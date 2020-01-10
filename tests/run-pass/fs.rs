// ignore-windows: File handling is not implemented yet
// compile-flags: -Zmiri-disable-isolation

use std::fs::{File, remove_file};
use std::io::{Read, Write, ErrorKind, Result};
use std::path::{PathBuf, Path};

fn test_metadata(bytes: &[u8], path: &Path) -> Result<()> {
    // Test that the file metadata is correct.
    let metadata = path.metadata()?;
    // `path` should point to a file.
    assert!(metadata.is_file());
    // The size of the file must be equal to the number of written bytes.
    assert_eq!(bytes.len() as u64, metadata.len());
    Ok(())
}

fn main() {
    let tmp = std::env::temp_dir();
    let filename = PathBuf::from("miri_test_fs.txt");
    let path = tmp.join(&filename);
    let symlink_path = tmp.join("miri_test_fs_symlink.txt");
    let bytes = b"Hello, World!\n";
    // Clean the paths for robustness.
    remove_file(&path).ok();
    remove_file(&symlink_path).ok();

    // Test creating, writing and closing a file (closing is tested when `file` is dropped).
    let mut file = File::create(&path).unwrap();
    // Writing 0 bytes should not change the file contents.
    file.write(&mut []).unwrap();

    file.write(bytes).unwrap();
    // Test opening, reading and closing a file.
    let mut file = File::open(&path).unwrap();
    let mut contents = Vec::new();
    // Reading 0 bytes should not move the file pointer.
    file.read(&mut []).unwrap();
    // Reading until EOF should get the whole text.
    file.read_to_end(&mut contents).unwrap();
    assert_eq!(bytes, contents.as_slice());

    // Test that metadata of an absolute path is correct.
    test_metadata(bytes, &path).unwrap();
    // Test that metadata of a relative path is correct.
    std::env::set_current_dir(&tmp).unwrap();
    test_metadata(bytes, &filename).unwrap();

    // Creating a symbolic link should succeed.
    std::os::unix::fs::symlink(&path, &symlink_path).unwrap();
    // Test that the symbolic link has the same contents as the file.
    let mut symlink_file = File::open(&symlink_path).unwrap();
    let mut contents = Vec::new();
    symlink_file.read_to_end(&mut contents).unwrap();
    assert_eq!(bytes, contents.as_slice());
    // Test that metadata of a symbolic link is correct.
    test_metadata(bytes, &symlink_path).unwrap();
    // Test that the metadata of a symbolic link is correct when not following it.
    assert!(symlink_path.symlink_metadata().unwrap().file_type().is_symlink());
    // Removing symbolic link should succeed.
    remove_file(&symlink_path).unwrap();

    // Removing file should succeed.
    remove_file(&path).unwrap();

    // The two following tests also check that the `__errno_location()` shim is working properly.
    // Opening a non-existing file should fail with a "not found" error.
    assert_eq!(ErrorKind::NotFound, File::open(&path).unwrap_err().kind());
    // Removing a non-existing file should fail with a "not found" error.
    assert_eq!(ErrorKind::NotFound, remove_file(&path).unwrap_err().kind());
    // Reading the metadata of a non-existing file should fail with a "not found" error.
    assert_eq!(ErrorKind::NotFound, test_metadata(bytes, &path).unwrap_err().kind());
}
