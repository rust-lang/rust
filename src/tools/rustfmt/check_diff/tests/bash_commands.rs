use check_diff::change_directory_to_path;
use std::env;
use tempfile::Builder;

#[test]
fn cd_test() {
    // Creates an empty directory in the current working directory
    let dir = Builder::new().tempdir_in("").unwrap();
    let dest_path = dir.path();
    change_directory_to_path(dest_path).unwrap();
    assert_eq!(env::current_dir().unwrap(), dest_path);
}
