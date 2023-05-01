//@ignore-target-windows: File handling is not implemented yet
//@compile-flags: -Zmiri-disable-isolation

#![feature(io_error_more)]
#![feature(io_error_uncategorized)]

use std::collections::HashMap;
use std::ffi::{c_char, OsString};
use std::fs::{
    canonicalize, create_dir, read_dir, read_link, remove_dir, remove_dir_all, remove_file, rename,
    File, OpenOptions,
};
use std::io::{Error, ErrorKind, IsTerminal, Read, Result, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

fn main() {
    test_path_conversion();
    test_file();
    test_file_clone();
    test_file_create_new();
    test_seek();
    test_metadata();
    test_file_set_len();
    test_file_sync();
    test_symlink();
    test_errors();
    test_rename();
    test_directory();
    test_canonicalize();
    test_from_raw_os_error();
}

fn host_to_target_path(path: String) -> PathBuf {
    use std::ffi::{CStr, CString};

    let path = CString::new(path).unwrap();
    let mut out = Vec::with_capacity(1024);

    unsafe {
        extern "Rust" {
            fn miri_host_to_target_path(
                path: *const c_char,
                out: *mut c_char,
                out_size: usize,
            ) -> usize;
        }
        let ret = miri_host_to_target_path(path.as_ptr(), out.as_mut_ptr(), out.capacity());
        assert_eq!(ret, 0);
        let out = CStr::from_ptr(out.as_ptr()).to_str().unwrap();
        PathBuf::from(out)
    }
}

fn tmp() -> PathBuf {
    let path = std::env::var("MIRI_TEMP")
        .unwrap_or_else(|_| std::env::temp_dir().into_os_string().into_string().unwrap());
    // These are host paths. We need to convert them to the target.
    host_to_target_path(path)
}

/// Prepare: compute filename and make sure the file does not exist.
fn prepare(filename: &str) -> PathBuf {
    let path = tmp().join(filename);
    // Clean the paths for robustness.
    remove_file(&path).ok();
    path
}

/// Prepare directory: compute directory name and make sure it does not exist.
fn prepare_dir(dirname: &str) -> PathBuf {
    let path = tmp().join(&dirname);
    // Clean the directory for robustness.
    remove_dir_all(&path).ok();
    path
}

/// Prepare like above, and also write some initial content to the file.
fn prepare_with_content(filename: &str, content: &[u8]) -> PathBuf {
    let path = prepare(filename);
    let mut file = File::create(&path).unwrap();
    file.write(content).unwrap();
    path
}

fn test_path_conversion() {
    let tmp = tmp();
    assert!(tmp.is_absolute(), "{:?} is not absolute", tmp);
    assert!(tmp.is_dir(), "{:?} is not a directory", tmp);
}

fn test_file() {
    let bytes = b"Hello, World!\n";
    let path = prepare("miri_test_fs_file.txt");

    // Test creating, writing and closing a file (closing is tested when `file` is dropped).
    let mut file = File::create(&path).unwrap();
    // Writing 0 bytes should not change the file contents.
    file.write(&mut []).unwrap();
    assert_eq!(file.metadata().unwrap().len(), 0);

    file.write(bytes).unwrap();
    assert_eq!(file.metadata().unwrap().len(), bytes.len() as u64);
    // Test opening, reading and closing a file.
    let mut file = File::open(&path).unwrap();
    let mut contents = Vec::new();
    // Reading 0 bytes should not move the file pointer.
    file.read(&mut []).unwrap();
    // Reading until EOF should get the whole text.
    file.read_to_end(&mut contents).unwrap();
    assert_eq!(bytes, contents.as_slice());

    assert!(!file.is_terminal());

    // Removing file should succeed.
    remove_file(&path).unwrap();
}

fn test_file_clone() {
    let bytes = b"Hello, World!\n";
    let path = prepare_with_content("miri_test_fs_file_clone.txt", bytes);

    // Cloning a file should be successful.
    let file = File::open(&path).unwrap();
    let mut cloned = file.try_clone().unwrap();
    // Reading from a cloned file should get the same text.
    let mut contents = Vec::new();
    cloned.read_to_end(&mut contents).unwrap();
    assert_eq!(bytes, contents.as_slice());

    // Removing file should succeed.
    remove_file(&path).unwrap();
}

fn test_file_create_new() {
    let path = prepare("miri_test_fs_file_create_new.txt");

    // Creating a new file that doesn't yet exist should succeed.
    OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
    // Creating a new file that already exists should fail.
    assert_eq!(
        ErrorKind::AlreadyExists,
        OpenOptions::new().write(true).create_new(true).open(&path).unwrap_err().kind()
    );
    // Optionally creating a new file that already exists should succeed.
    OpenOptions::new().write(true).create(true).open(&path).unwrap();

    // Clean up
    remove_file(&path).unwrap();
}

fn test_seek() {
    let bytes = b"Hello, entire World!\n";
    let path = prepare_with_content("miri_test_fs_seek.txt", bytes);

    let mut file = File::open(&path).unwrap();
    let mut contents = Vec::new();
    file.read_to_end(&mut contents).unwrap();
    assert_eq!(bytes, contents.as_slice());
    // Test that seeking to the beginning and reading until EOF gets the text again.
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut contents = Vec::new();
    file.read_to_end(&mut contents).unwrap();
    assert_eq!(bytes, contents.as_slice());
    // Test seeking relative to the end of the file.
    file.seek(SeekFrom::End(-1)).unwrap();
    let mut contents = Vec::new();
    file.read_to_end(&mut contents).unwrap();
    assert_eq!(&bytes[bytes.len() - 1..], contents.as_slice());
    // Test seeking relative to the current position.
    file.seek(SeekFrom::Start(5)).unwrap();
    file.seek(SeekFrom::Current(-3)).unwrap();
    let mut contents = Vec::new();
    file.read_to_end(&mut contents).unwrap();
    assert_eq!(&bytes[2..], contents.as_slice());

    // Removing file should succeed.
    remove_file(&path).unwrap();
}

fn check_metadata(bytes: &[u8], path: &Path) -> Result<()> {
    // Test that the file metadata is correct.
    let metadata = path.metadata()?;
    // `path` should point to a file.
    assert!(metadata.is_file());
    // The size of the file must be equal to the number of written bytes.
    assert_eq!(bytes.len() as u64, metadata.len());
    Ok(())
}

fn test_metadata() {
    let bytes = b"Hello, meta-World!\n";
    let path = prepare_with_content("miri_test_fs_metadata.txt", bytes);

    // Test that metadata of an absolute path is correct.
    check_metadata(bytes, &path).unwrap();
    // Test that metadata of a relative path is correct.
    std::env::set_current_dir(path.parent().unwrap()).unwrap();
    check_metadata(bytes, Path::new(path.file_name().unwrap())).unwrap();

    // Removing file should succeed.
    remove_file(&path).unwrap();
}

fn test_file_set_len() {
    let bytes = b"Hello, World!\n";
    let path = prepare_with_content("miri_test_fs_set_len.txt", bytes);

    // Test extending the file
    let mut file = OpenOptions::new().read(true).write(true).open(&path).unwrap();
    let bytes_extended = b"Hello, World!\n\x00\x00\x00\x00\x00\x00";
    file.set_len(20).unwrap();
    let mut contents = Vec::new();
    file.read_to_end(&mut contents).unwrap();
    assert_eq!(bytes_extended, contents.as_slice());

    // Test truncating the file
    file.seek(SeekFrom::Start(0)).unwrap();
    file.set_len(10).unwrap();
    let mut contents = Vec::new();
    file.read_to_end(&mut contents).unwrap();
    assert_eq!(&bytes[..10], contents.as_slice());

    // Can't use set_len on a file not opened for writing
    let file = OpenOptions::new().read(true).open(&path).unwrap();
    assert_eq!(ErrorKind::InvalidInput, file.set_len(14).unwrap_err().kind());

    remove_file(&path).unwrap();
}

fn test_file_sync() {
    let bytes = b"Hello, World!\n";
    let path = prepare_with_content("miri_test_fs_sync.txt", bytes);

    // Test that we can call sync_data and sync_all (can't readily test effects of this operation)
    let file = OpenOptions::new().write(true).open(&path).unwrap();
    file.sync_data().unwrap();
    file.sync_all().unwrap();

    // Test that we can call sync_data and sync_all on a file opened for reading.
    let file = File::open(&path).unwrap();
    file.sync_data().unwrap();
    file.sync_all().unwrap();

    remove_file(&path).unwrap();
}

fn test_symlink() {
    let bytes = b"Hello, World!\n";
    let path = prepare_with_content("miri_test_fs_link_target.txt", bytes);
    let symlink_path = prepare("miri_test_fs_symlink.txt");

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

fn test_errors() {
    let bytes = b"Hello, World!\n";
    let path = prepare("miri_test_fs_errors.txt");

    // The following tests also check that the `__errno_location()` shim is working properly.
    // Opening a non-existing file should fail with a "not found" error.
    assert_eq!(ErrorKind::NotFound, File::open(&path).unwrap_err().kind());
    // Make sure we can also format this.
    format!("{0:?}: {0}", File::open(&path).unwrap_err());
    // Removing a non-existing file should fail with a "not found" error.
    assert_eq!(ErrorKind::NotFound, remove_file(&path).unwrap_err().kind());
    // Reading the metadata of a non-existing file should fail with a "not found" error.
    assert_eq!(ErrorKind::NotFound, check_metadata(bytes, &path).unwrap_err().kind());
}

fn test_rename() {
    // Renaming a file should succeed.
    let path1 = prepare("miri_test_fs_rename_source.txt");
    let path2 = prepare("miri_test_fs_rename_destination.txt");

    let file = File::create(&path1).unwrap();
    drop(file);

    // Renaming should succeed
    rename(&path1, &path2).unwrap();
    // Check that the old file path isn't present
    assert_eq!(ErrorKind::NotFound, path1.metadata().unwrap_err().kind());
    // Check that the file has moved successfully
    assert!(path2.metadata().unwrap().is_file());

    // Renaming a nonexistent file should fail
    assert_eq!(ErrorKind::NotFound, rename(&path1, &path2).unwrap_err().kind());

    remove_file(&path2).unwrap();
}

fn test_canonicalize() {
    let dir_path = prepare_dir("miri_test_fs_dir");
    create_dir(&dir_path).unwrap();
    let path = dir_path.join("test_file");
    drop(File::create(&path).unwrap());

    let p = canonicalize(format!("{}/./test_file", dir_path.to_string_lossy())).unwrap();
    assert_eq!(p.to_string_lossy().find('.'), None);

    remove_dir_all(&dir_path).unwrap();
}

fn test_directory() {
    let dir_path = prepare_dir("miri_test_fs_dir");
    // Creating a directory should succeed.
    create_dir(&dir_path).unwrap();
    // Test that the metadata of a directory is correct.
    assert!(dir_path.metadata().unwrap().is_dir());
    // Creating a directory when it already exists should fail.
    assert_eq!(ErrorKind::AlreadyExists, create_dir(&dir_path).unwrap_err().kind());

    // Create some files and dirs inside the directory
    let path_1 = dir_path.join("test_file_1");
    drop(File::create(&path_1).unwrap());
    let path_2 = dir_path.join("test_file_2");
    drop(File::create(&path_2).unwrap());
    let dir_1 = dir_path.join("test_dir_1");
    create_dir(&dir_1).unwrap();
    // Test that read_dir metadata calls succeed
    assert_eq!(
        HashMap::from([
            (OsString::from("test_file_1"), true),
            (OsString::from("test_file_2"), true),
            (OsString::from("test_dir_1"), false)
        ]),
        read_dir(&dir_path)
            .unwrap()
            .map(|e| {
                let e = e.unwrap();
                (e.file_name(), e.metadata().unwrap().is_file())
            })
            .collect::<HashMap<_, _>>()
    );
    // Deleting the directory should fail, since it is not empty.
    assert_eq!(ErrorKind::DirectoryNotEmpty, remove_dir(&dir_path).unwrap_err().kind());
    // Clean up the files in the directory
    remove_file(&path_1).unwrap();
    remove_file(&path_2).unwrap();
    remove_dir(&dir_1).unwrap();
    // Now there should be nothing left in the directory.
    let dir_iter = read_dir(&dir_path).unwrap();
    let file_names = dir_iter.map(|e| e.unwrap().file_name()).collect::<Vec<_>>();
    assert!(file_names.is_empty());

    // Deleting the directory should succeed.
    remove_dir(&dir_path).unwrap();
    // Reading the metadata of a nonexistent directory should fail with a "not found" error.
    assert_eq!(ErrorKind::NotFound, check_metadata(&[], &dir_path).unwrap_err().kind());

    // To test remove_dir_all, re-create the directory with a file and a directory in it.
    create_dir(&dir_path).unwrap();
    drop(File::create(&path_1).unwrap());
    create_dir(&path_2).unwrap();
    remove_dir_all(&dir_path).unwrap();
}

fn test_from_raw_os_error() {
    let code = 6; // not a code that std or Miri know
    let error = Error::from_raw_os_error(code);
    assert!(matches!(error.kind(), ErrorKind::Uncategorized));
    // Make sure we can also format this.
    format!("{error:?}");
}
