use std::env;
use std::path::PathBuf;

fn main() {
    println!("subcrate testing");

    let env_dir = env::current_dir().unwrap();
    let crate_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    // CWD should be crate root.
    assert_eq!(env_dir, crate_dir);
}
