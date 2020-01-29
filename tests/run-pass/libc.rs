// ignore-windows: No libc on Windows
// compile-flags: -Zmiri-disable-isolation

#![feature(rustc_private)]

#[allow(unused)] // necessary on macos due to conditional compilation
extern crate libc;

#[cfg(not(target_os = "macos"))]
fn test_posix_fadvise() {
    use std::convert::TryInto;
    use std::env::temp_dir;
    use std::fs::{File, remove_file};
    use std::io::Write;
    use std::os::unix::io::AsRawFd;

    let path = temp_dir().join("miri_test_libc.txt");
    // Cleanup before test
    remove_file(&path).ok();

    // Set up an open file
    let mut file = File::create(&path).unwrap();
    let bytes = b"Hello, World!\n";
    file.write(bytes).unwrap();

    // Test calling posix_fadvise on a file.
    let result = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            0,
            bytes.len().try_into().unwrap(),
            libc::POSIX_FADV_DONTNEED,
        )
    };
    drop(file);
    remove_file(&path).unwrap();
    assert_eq!(result, 0);
}

fn main() {
    #[cfg(not(target_os = "macos"))]
    test_posix_fadvise();
}
