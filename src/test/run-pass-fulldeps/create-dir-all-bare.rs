use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let path = PathBuf::from(env::var_os("RUST_TEST_TMPDIR").unwrap());
    env::set_current_dir(&path).unwrap();
    fs::create_dir_all("create-dir-all-bare").unwrap();
}
