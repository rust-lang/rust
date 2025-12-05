// Symlink tests are separate since they don't in general work on a Windows host.
//@ignore-host: windows # creating symlinks requires admin permissions on Windows
//@ignore-target: windows # File handling is not implemented yet
//@compile-flags: -Zmiri-disable-isolation

use std::fs::{File, read_link, remove_file};
use std::io::{Read, Result};
use std::path::Path;

#[path = "../../utils/mod.rs"]
mod utils;

fn check_metadata(bytes: &[u8], path: &Path) -> Result<()> {
    // Test that the file metadata is correct.
    let metadata = path.metadata()?;
    // `path` should point to a file.
    assert!(metadata.is_file());
    // The size of the file must be equal to the number of written bytes.
    assert_eq!(bytes.len() as u64, metadata.len());
    Ok(())
}

fn main() {
    let bytes = b"Hello, World!\n";
    let path = utils::prepare_with_content("miri_test_fs_link_target.txt", bytes);
    let symlink_path = utils::prepare("miri_test_fs_symlink.txt");

    // Creating a symbolic link should succeed.
    #[cfg(unix)]
    std::os::unix::fs::symlink(&path, &symlink_path).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(&path, &symlink_path).unwrap();
    // Test that the symbolic link has the same contents as the file.
    let mut symlink_file = File::open(&symlink_path).unwrap();
    let mut contents = Vec::new();
    symlink_file.read_to_end(&mut contents).unwrap();
    assert_eq!(bytes, contents.as_slice());

    // Test that metadata of a symbolic link (i.e., the file it points to) is correct.
    check_metadata(bytes, &symlink_path).unwrap();
    // Test that the metadata of a symbolic link is correct when not following it.
    assert!(symlink_path.symlink_metadata().unwrap().file_type().is_symlink());
    // Check that we can follow the link.
    assert_eq!(read_link(&symlink_path).unwrap(), path);
    // Removing symbolic link should succeed.
    remove_file(&symlink_path).unwrap();

    // Removing file should succeed.
    remove_file(&path).unwrap();
}
