use std::env;
use std::path::PathBuf;

fn main() {
    println!("subcrate running");

    // CWD should be workspace root, i.e., one level up from crate root.
    // We have to normalize slashes, as the env var might be set for a different target's conventions.
    let env_dir = env::current_dir().unwrap();
    let env_dir = env_dir.to_string_lossy().replace("\\", "/");
    let crate_dir = env::var_os("CARGO_MANIFEST_DIR").unwrap();
    let crate_dir = crate_dir.to_string_lossy().replace("\\", "/");
    let crate_dir = PathBuf::from(crate_dir);
    let crate_dir = crate_dir.parent().unwrap().to_string_lossy();
    assert_eq!(env_dir, crate_dir);
}
