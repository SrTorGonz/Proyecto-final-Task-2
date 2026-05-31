import OpenVisus as ov

for var in ['u', 'v', 'p', 't']:
    url = f"https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/GEOS/GEOS_{var.upper()}/{var.lower()}_face_0_depth_52_time_0_10269.idx"
    try:
        db = ov.LoadDataset(url)
        print(f"Var: {var.upper()}")
        print(f"  Dimensions: {db.getLogicBox()[1][0]}*{db.getLogicBox()[1][1]}*{db.getLogicBox()[1][2]}")
        print(f"  Total Timesteps: {len(db.getTimesteps())}")
        print(f"  Field: {db.getField().name}")
        print(f"  Field DataType: {db.getField().dtype.toString()}")
    except Exception as e:
        print(f"Failed to load {var.upper()}: {e}")
