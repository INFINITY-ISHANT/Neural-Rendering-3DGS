import argparse
import sys
from pathlib import Path

def run_colmap(dataset_path: str):
    try:
        import pycolmap
    except ImportError:
        print("Error: pycolmap is not installed. Please run: pip install pycolmap")
        sys.exit(1)

    dataset_path = Path(dataset_path).resolve()
    image_dir = dataset_path / "images"
    
    if not image_dir.exists():
        print(f"Error: Could not find 'images' directory at {image_dir}")
        print("Please ensure your dataset is structured as:")
        print("  <dataset_dir>/")
        print("  └── images/")
        print("      ├── image1.jpg")
        print("      ├── image2.jpg")
        print("      └── ...")
        sys.exit(1)

    # 3DGS implementations strictly expect the sparse model in <dataset_dir>/sparse/0/
    sparse_path = dataset_path / "sparse" / "0"
    sparse_path.mkdir(parents=True, exist_ok=True)
    
    database_path = dataset_path / "database.db"

    print(f"--- Starting COLMAP processing for {dataset_path} ---")
    
    print("1/3: Extracting features...")
    # Extract SIFT features using pycolmap
    pycolmap.extract_features(database_path, image_dir)

    print("2/3: Matching features...")
    # Exhaustive matching is highly accurate but scales O(N^2). 
    # Use match_spatial() or match_vocab_tree() for datasets > 500 images.
    pycolmap.match_exhaustive(database_path)

    print("3/3: Incremental mapping (Structure from Motion)...")
    # This generates the point cloud and camera poses
    maps = pycolmap.incremental_mapping(database_path, image_dir, sparse_path)

    if maps:
        # Output is a dict of reconstructions. The first one (0) is usually the largest/best.
        best_reconstruction = maps[0]
        
        # Write cameras.bin, images.bin, and points3D.bin to sparse/0/
        best_reconstruction.write(sparse_path)
        
        print("\n SUCCESS! Sparse reconstruction completed.")
        print(f"Results saved to: {sparse_path}")
        print(f"You can now point your gsplat implementation to '{dataset_path}'")
    else:
        print("\n FAILURE: COLMAP could not reconstruct any models from the provided images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP sparse reconstruction for Gaussian Splatting.")
    parser.add_argument(
        "--dataset", 
        "-d",
        type=str, 
        required=True, 
        help="Path to the dataset directory containing an 'images' subfolder."
    )
    args = parser.parse_args()
    
    run_colmap(args.dataset)
