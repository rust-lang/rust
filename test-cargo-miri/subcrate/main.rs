use std::env;
use std::path::PathBuf;

fn main() {
    println!("subcrate running");

    let env_dir = env::current_dir().unwrap();
    let crate_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    // CWD should be workspace root, i.e., one level up from crate root.
    assert_eq!(env_dir, crate_dir.parent().unwrap());
}
