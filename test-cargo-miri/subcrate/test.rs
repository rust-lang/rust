use std::env;

use byteorder::{ByteOrder, LittleEndian};

fn main() {
    println!("subcrate testing");

    // CWD should be crate root.
    // We have to normalize slashes, as the env var might be set for a different target's conventions.
    let env_dir = env::current_dir().unwrap();
    let env_dir = env_dir.to_string_lossy().replace("\\", "/");
    let crate_dir = env::var_os("CARGO_MANIFEST_DIR").unwrap();
    let crate_dir = crate_dir.to_string_lossy().replace("\\", "/");
    assert_eq!(env_dir, crate_dir);

    // Make sure we can call dev-dependencies.
    let _n = <LittleEndian as ByteOrder>::read_u32(&[1, 2, 3, 4]);
}
