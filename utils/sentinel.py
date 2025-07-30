#utils/sentinel.py
import requests, os, ee, json, datetime
from models.predictor import predict_nutrients

# Load service account key JSON from env variable as string
key_data_str = os.getenv("GEE_SERVICE_ACCOUNT_JSON")

if not key_data_str:
    raise ValueError("Missing GEE_SERVICE_ACCOUNT_JSON environment variable")

# Extract the client_email from the JSON string
key_dict = json.loads(key_data_str)
email = key_dict["client_email"]

# Authenticate with Earth Engine
credentials = ee.ServiceAccountCredentials(
    email=email,
    key_data=key_data_str  # This must be a JSON string
)

ee.Initialize(credentials)

# ===============================
# Function: get_satellite_features
# ===============================
def get_satellite_features(lat, lon, date_str):
    """
    Retrieves satellite features (bands and indices) from Sentinel-2
    for a given latitude, longitude, and date.

    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        date_str (str): Date in "YYYY-MM-DD" format.

    Returns:
        dict: A dictionary containing the satellite features.

    Raises:
        ValueError: If no suitable Sentinel-2 image is found or no data at the point.
    """
    # Parse the date string into a datetime object
    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    start_date = date.strftime("%Y-%m-%d")
    # End date is one day after the start date to define a 24-hour window
    end_date = (date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # Define the point of interest as an Earth Engine Geometry object
    point = ee.Geometry.Point([lon, lat])

    # Filter the Sentinel-2 Surface Reflectance Harmonized image collection
    # by bounds, date, and cloud percentage
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) # Filter for images with less than 20% cloud cover
    )

    # Define the list of required Sentinel-2 bands
    required_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B11", "B12"]

    # Function to check if an image contains all required bands
    # Using containsAll is more idiomatic and robust for this check
    def has_all_bands(image):
        return image.bandNames().containsAll(ee.List(required_bands))

    # Apply the band presence filter to the image collection
    # First, map a property 'has_bands' to each image, then filter based on it
    valid_images = collection \
        .map(lambda img: img.set("has_bands", has_all_bands(img))) \
        .filterMetadata("has_bands", "equals", True) # Changed to True as containsAll returns a boolean

    # Get the first image from the filtered collection.
    # If no images are found, .first() will return None (in Earth Engine context).
    image = valid_images.first()

    # Check if a valid image was found. If not, raise an error.
    if image is None:
        raise ValueError(
            f"No Sentinel-2 image with required bands found for location ({lat}, {lon}) "
            f"and date {date_str} (cloud cover < 20%)."
        )

    # Select only the required bands from the found image for further calculations
    selected_bands_image = image.select(required_bands)

    # Extract individual bands for use in expressions (improves readability)
    B1 = selected_bands_image.select("B1")
    B2 = selected_bands_image.select("B2")
    B3 = selected_bands_image.select("B3")
    B4 = selected_bands_image.select("B4")
    B5 = selected_bands_image.select("B5")
    B6 = selected_bands_image.select("B6")
    B7 = selected_bands_image.select("B7")
    B8 = selected_bands_image.select("B8")
    B9 = selected_bands_image.select("B9")
    B11 = selected_bands_image.select("B11")
    B12 = selected_bands_image.select("B12")

    # Calculate various vegetation and water indices
    NDVI_G = selected_bands_image.normalizedDifference(["B8", "B3"]).rename("NDVI_G")
    NDWI = selected_bands_image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    PSRI = selected_bands_image.expression("(B4 - B2) / B6", {"B2": B2, "B4": B4, "B6": B6}).rename("PSRI")
    TBVI1 = selected_bands_image.expression("(B6 + 0.5 * B5 - 0.5 * B2) / 2", {"B2": B2, "B5": B5, "B6": B6}).rename("TBVI1")
    NDVIRE1n = selected_bands_image.normalizedDifference(["B8", "B5"]).rename("NDVIRE1n")
    NDVIRE2n = selected_bands_image.normalizedDifference(["B8", "B6"]).rename("NDVIRE2n")
    NDVIRE3n = selected_bands_image.normalizedDifference(["B8", "B7"]).rename("NDVIRE3n")
    SR_n2 = selected_bands_image.expression("B8 / B4", {"B8": B8, "B4": B4}).rename("SR_n2")
    SR_N = selected_bands_image.expression("B8 / B5", {"B8": B8, "B5": B5}).rename("SR_N")
    BI = selected_bands_image.expression("sqrt(B11**2 + B12**2) / 2", {"B11": B11, "B12": B12}).rename("BI")
    CI = NDVIRE2n.rename("CI") # CI is directly assigned NDVIRE2n, ensure it's renamed
    SI = selected_bands_image.expression("B11 / B12", {"B11": B11, "B12": B12}).rename("SI")
    B8_minus_B4 = selected_bands_image.expression("B8 - B4", {"B8": B8, "B4": B4}).rename("B8_minus_B4")
    NDVI_G_times_PSRI = NDVI_G.multiply(PSRI).rename("NDVI_G_times_PSRI")

    # Add all calculated indices as new bands to the image
    all_bands = selected_bands_image.addBands([
        NDVI_G, NDWI, PSRI, TBVI1,
        NDVIRE1n, NDVIRE2n, NDVIRE3n,
        SR_n2, SR_N, BI, CI, SI,
        B8_minus_B4, NDVI_G_times_PSRI
    ])

    # Sample the image at the defined point with a scale of 10 meters
    # .first() is used to get the properties of the feature
    sampled = all_bands.sample(point, scale=10).first().getInfo()

    # Check if data was successfully sampled at the point
    if not sampled:
        raise ValueError("No data found at the given point and date after sampling.")

    # Extract the properties (band values and index values) from the sampled data
    props = sampled["properties"]

    # Return a dictionary containing the original coordinates and all retrieved features
    return {
        "latitude": lat, "longitude": lon,
        "B1": props.get("B1"), "B2": props.get("B2"), "B3": props.get("B3"), "B4": props.get("B4"),
        "B5": props.get("B5"), "B6": props.get("B6"), "B7": props.get("B7"), "B8": props.get("B8"),
        "B9": props.get("B9"), "B11": props.get("B11"), "B12": props.get("B12"),
        "NDVI_G": props.get("NDVI_G"), "NDWI": props.get("NDWI"), "PSRI": props.get("PSRI"),
        "TBVI1": props.get("TBVI1"), "NDVIRE1n": props.get("NDVIRE1n"),
        "NDVIRE2n": props.get("NDVIRE2n"), "NDVIRE3n": props.get("NDVIRE3n"),
        "SR_n2": props.get("SR_n2"), "SR_N": props.get("SR_N"), "BI": props.get("BI"),
        "CI": props.get("CI"), # CI is already NDVIRE2n, ensure it's retrieved by its new name
        "SI": props.get("SI"),
        "B8_minus_B4": props.get("B8_minus_B4"),
        "NDVI_G_times_PSRI": props.get("NDVI_G_times_PSRI")
    }


def get_satellite_mean_nutrient(lat, lon, date_str, nutrients):
    ref_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    start_date = ref_date - datetime.timedelta(weeks=52)
    point = ee.Geometry.Point([lon, lat])

    # Function to add indices
    def add_indices(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI_G')
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        psri = image.expression('(B4 - B2) / B6', {
            'B4': image.select('B4'), 'B2': image.select('B2'), 'B6': image.select('B6')
        }).rename('PSRI')
        tbvi1 = image.expression('(B6 - B5) / (B6 + B5)', {
            'B6': image.select('B6'), 'B5': image.select('B5')
        }).rename('TBVI1')
        ndvire1n = image.normalizedDifference(['B5', 'B4']).rename('NDVIRE1n')
        ndvire2n = image.normalizedDifference(['B6', 'B4']).rename('NDVIRE2n')
        ndvire3n = image.normalizedDifference(['B7', 'B4']).rename('NDVIRE3n')
        sr_n2 = image.expression('B8 / B3', {'B8': image.select('B8'), 'B3': image.select('B3')}).rename('SR_n2')
        sr_n = image.expression('B8 / B2', {'B8': image.select('B8'), 'B2': image.select('B2')}).rename('SR_N')
        bi = image.expression('(B11 + B12) / B8', {
            'B11': image.select('B11'), 'B12': image.select('B12'), 'B8': image.select('B8')
        }).rename('BI')
        ci = ndvire2n.rename('CI')
        si = image.expression('B11 / B12', {
            'B11': image.select('B11'), 'B12': image.select('B12')
        }).rename('SI')
        b8_minus_b4 = image.expression('B8 - B4', {
            'B8': image.select('B8'), 'B4': image.select('B4')
        }).rename('B8_minus_B4')
        ndvi_times_psri = ndvi.multiply(psri).rename('NDVI_G_times_PSRI')

        return image.addBands([
            ndvi, ndwi, psri, tbvi1,
            ndvire1n, ndvire2n, ndvire3n,
            sr_n2, sr_n, bi, ci, si,
            b8_minus_b4, ndvi_times_psri
        ])

    # Prepare collection
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B11", "B12"]
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(point) \
        .filterDate(start_date.strftime("%Y-%m-%d"), date_str) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
        .map(add_indices)

    # Generate weekly dates
    def generate_weeks(start, end):
        weeks = []
        while start <= end:
            weeks.append(start)
            start += datetime.timedelta(days=7)
        return weeks

    weekly_dates = generate_weeks(start_date, ref_date)
    nutrient_values = {nutrient: [] for nutrient in nutrients}

    for week_start in weekly_dates:
        week_end = week_start + datetime.timedelta(days=6)
        try:
            weekly_images = collection.filterDate(week_start.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d"))

            band_list = bands + [
                "NDVI_G", "NDWI", "PSRI", "TBVI1", "NDVIRE1n", "NDVIRE2n", "NDVIRE3n",
                "SR_n2", "SR_N", "BI", "CI", "SI", "B8_minus_B4", "NDVI_G_times_PSRI"
            ]
            mean_dict = weekly_images.select(band_list).mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=10,
                maxPixels=1e9
            ).getInfo()

            if mean_dict and "B4" in mean_dict:
                props = {k: float(mean_dict[k]) for k in band_list if mean_dict.get(k) is not None}
                props["latitude"] = lat
                props["longitude"] = lon

                prediction = predict_nutrients({
                    "nutrients": nutrients,
                    "data": props
                })

                for nutrient in nutrients:
                    nutrient_values[nutrient].append(prediction[nutrient])
        except Exception as e:
            print(f"⚠️ Week {week_start.strftime('%Y-%m-%d')} skipped: {e}")

    # Compute mean prediction for each nutrient
    final_result = {
        nutrient: round(sum(vals) / len(vals), 3) if vals else None
        for nutrient, vals in nutrient_values.items()
    }

    final_result.update({
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": date_str,
        "weeks_count": len(weekly_dates)
    })

    return final_result
