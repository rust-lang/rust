//@ignore-target-windows: No libc on Windows
//@compile-flags: -Zmiri-disable-isolation
#![feature(io_error_more)]

use std::fs::{remove_file, File};
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;

fn tmp() -> PathBuf {
    use std::ffi::{c_char, CStr, CString};

    let path = std::env::var("MIRI_TEMP")
        .unwrap_or_else(|_| std::env::temp_dir().into_os_string().into_string().unwrap());
    // These are host paths. We need to convert them to the target.
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

/// Test allocating variant of `realpath`.
fn test_posix_realpath_alloc() {
    use std::ffi::OsString;
    use std::ffi::{CStr, CString};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::ffi::OsStringExt;

    let buf;
    let path = tmp().join("miri_test_libc_posix_realpath_alloc");
    let c_path = CString::new(path.as_os_str().as_bytes()).expect("CString::new failed");

    // Cleanup before test.
    remove_file(&path).ok();
    // Create file.
    drop(File::create(&path).unwrap());
    unsafe {
        let r = libc::realpath(c_path.as_ptr(), std::ptr::null_mut());
        assert!(!r.is_null());
        buf = CStr::from_ptr(r).to_bytes().to_vec();
        libc::free(r as *mut _);
    }
    let canonical = PathBuf::from(OsString::from_vec(buf));
    assert_eq!(path.file_name(), canonical.file_name());

    // Cleanup after test.
    remove_file(&path).unwrap();
}

/// Test non-allocating variant of `realpath`.
fn test_posix_realpath_noalloc() {
    use std::ffi::{CStr, CString};
    use std::os::unix::ffi::OsStrExt;

    let path = tmp().join("miri_test_libc_posix_realpath_noalloc");
    let c_path = CString::new(path.as_os_str().as_bytes()).expect("CString::new failed");

    let mut v = vec![0; libc::PATH_MAX as usize];

    // Cleanup before test.
    remove_file(&path).ok();
    // Create file.
    drop(File::create(&path).unwrap());
    unsafe {
        let r = libc::realpath(c_path.as_ptr(), v.as_mut_ptr());
        assert!(!r.is_null());
    }
    let c = unsafe { CStr::from_ptr(v.as_ptr()) };
    let canonical = PathBuf::from(c.to_str().expect("CStr to str"));

    assert_eq!(path.file_name(), canonical.file_name());

    // Cleanup after test.
    remove_file(&path).unwrap();
}

/// Test failure cases for `realpath`.
fn test_posix_realpath_errors() {
    use std::ffi::CString;
    use std::io::ErrorKind;

    // Test non-existent path returns an error.
    let c_path = CString::new("./nothing_to_see_here").expect("CString::new failed");
    let r = unsafe { libc::realpath(c_path.as_ptr(), std::ptr::null_mut()) };
    assert!(r.is_null());
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::ENOENT));
    assert_eq!(e.kind(), ErrorKind::NotFound);
}

#[cfg(target_os = "linux")]
fn test_posix_fadvise() {
    use std::convert::TryInto;
    use std::io::Write;

    let path = tmp().join("miri_test_libc_posix_fadvise.txt");
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

#[cfg(target_os = "linux")]
fn test_sync_file_range() {
    use std::io::Write;

    let path = tmp().join("miri_test_libc_sync_file_range.txt");
    // Cleanup before test.
    remove_file(&path).ok();

    // Write to a file.
    let mut file = File::create(&path).unwrap();
    let bytes = b"Hello, World!\n";
    file.write(bytes).unwrap();

    // Test calling sync_file_range on the file.
    let result_1 = unsafe {
        libc::sync_file_range(
            file.as_raw_fd(),
            0,
            0,
            libc::SYNC_FILE_RANGE_WAIT_BEFORE
                | libc::SYNC_FILE_RANGE_WRITE
                | libc::SYNC_FILE_RANGE_WAIT_AFTER,
        )
    };
    drop(file);

    // Test calling sync_file_range on a file opened for reading.
    let file = File::open(&path).unwrap();
    let result_2 = unsafe {
        libc::sync_file_range(
            file.as_raw_fd(),
            0,
            0,
            libc::SYNC_FILE_RANGE_WAIT_BEFORE
                | libc::SYNC_FILE_RANGE_WRITE
                | libc::SYNC_FILE_RANGE_WAIT_AFTER,
        )
    };
    drop(file);

    remove_file(&path).unwrap();
    assert_eq!(result_1, 0);
    assert_eq!(result_2, 0);
}

/// Tests whether each thread has its own `__errno_location`.
fn test_thread_local_errno() {
    #[cfg(target_os = "linux")]
    use libc::__errno_location;
    #[cfg(any(target_os = "macos", target_os = "freebsd"))]
    use libc::__error as __errno_location;

    unsafe {
        *__errno_location() = 0xBEEF;
        std::thread::spawn(|| {
            assert_eq!(*__errno_location(), 0);
            *__errno_location() = 0xBAD1DEA;
            assert_eq!(*__errno_location(), 0xBAD1DEA);
        })
        .join()
        .unwrap();
        assert_eq!(*__errno_location(), 0xBEEF);
    }
}

/// Tests whether clock support exists at all
fn test_clocks() {
    let mut tp = std::mem::MaybeUninit::<libc::timespec>::uninit();
    let is_error = unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, tp.as_mut_ptr()) };
    assert_eq!(is_error, 0);
    let is_error = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, tp.as_mut_ptr()) };
    assert_eq!(is_error, 0);
    #[cfg(target_os = "linux")]
    {
        let is_error = unsafe { libc::clock_gettime(libc::CLOCK_REALTIME_COARSE, tp.as_mut_ptr()) };
        assert_eq!(is_error, 0);
        let is_error =
            unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_COARSE, tp.as_mut_ptr()) };
        assert_eq!(is_error, 0);
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let is_error = unsafe { libc::clock_gettime(libc::CLOCK_UPTIME_RAW, tp.as_mut_ptr()) };
        assert_eq!(is_error, 0);
    }
}

fn test_posix_gettimeofday() {
    let mut tp = std::mem::MaybeUninit::<libc::timeval>::uninit();
    let tz = std::ptr::null_mut::<libc::timezone>();
    #[cfg(target_os = "macos")] // `tz` has a different type on macOS
    let tz = tz as *mut libc::c_void;
    let is_error = unsafe { libc::gettimeofday(tp.as_mut_ptr(), tz) };
    assert_eq!(is_error, 0);
    let tv = unsafe { tp.assume_init() };
    assert!(tv.tv_sec > 0);
    assert!(tv.tv_usec >= 0); // Theoretically this could be 0.

    // Test that non-null tz returns an error.
    let mut tz = std::mem::MaybeUninit::<libc::timezone>::uninit();
    let tz_ptr = tz.as_mut_ptr();
    #[cfg(target_os = "macos")] // `tz` has a different type on macOS
    let tz_ptr = tz_ptr as *mut libc::c_void;
    let is_error = unsafe { libc::gettimeofday(tp.as_mut_ptr(), tz_ptr) };
    assert_eq!(is_error, -1);
}

fn test_isatty() {
    // Testing whether our isatty shim returns the right value would require controlling whether
    // these streams are actually TTYs, which is hard.
    // For now, we just check that these calls are supported at all.
    unsafe {
        libc::isatty(libc::STDIN_FILENO);
        libc::isatty(libc::STDOUT_FILENO);
        libc::isatty(libc::STDERR_FILENO);

        // But when we open a file, it is definitely not a TTY.
        let path = tmp().join("notatty.txt");
        // Cleanup before test.
        remove_file(&path).ok();
        let file = File::create(&path).unwrap();

        assert_eq!(libc::isatty(file.as_raw_fd()), 0);
        assert_eq!(std::io::Error::last_os_error().raw_os_error().unwrap(), libc::ENOTTY);

        // Cleanup after test.
        drop(file);
        remove_file(&path).unwrap();
    }
}

fn test_posix_mkstemp() {
    use std::ffi::CString;
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::FromRawFd;
    use std::path::Path;

    let valid_template = "fooXXXXXX";
    // C needs to own this as `mkstemp(3)` says:
    // "Since it will be modified, `template` must not be a string constant, but
    // should be declared as a character array."
    // There seems to be no `as_mut_ptr` on `CString` so we need to use `into_raw`.
    let ptr = CString::new(valid_template).unwrap().into_raw();
    let fd = unsafe { libc::mkstemp(ptr) };
    // Take ownership back in Rust to not leak memory.
    let slice = unsafe { CString::from_raw(ptr) };
    assert!(fd > 0);
    let osstr = OsStr::from_bytes(slice.to_bytes());
    let path: &Path = osstr.as_ref();
    let name = path.file_name().unwrap().to_string_lossy();
    assert!(name.ne("fooXXXXXX"));
    assert!(name.starts_with("foo"));
    assert_eq!(name.len(), 9);
    assert_eq!(
        name.chars().skip(3).filter(char::is_ascii_alphanumeric).collect::<Vec<char>>().len(),
        6
    );
    let file = unsafe { File::from_raw_fd(fd) };
    assert!(file.set_len(0).is_ok());

    let invalid_templates = vec!["foo", "barXX", "XXXXXXbaz", "whatXXXXXXever", "X"];
    for t in invalid_templates {
        let ptr = CString::new(t).unwrap().into_raw();
        let fd = unsafe { libc::mkstemp(ptr) };
        let _ = unsafe { CString::from_raw(ptr) };
        // "On error, -1 is returned, and errno is set to
        // indicate the error"
        assert_eq!(fd, -1);
        let e = std::io::Error::last_os_error();
        assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    }
}

fn main() {
    test_posix_gettimeofday();
    test_posix_mkstemp();

    test_posix_realpath_alloc();
    test_posix_realpath_noalloc();
    test_posix_realpath_errors();

    test_thread_local_errno();

    test_isatty();
    test_clocks();

    #[cfg(target_os = "linux")]
    {
        test_posix_fadvise();
        test_sync_file_range();
    }
}
