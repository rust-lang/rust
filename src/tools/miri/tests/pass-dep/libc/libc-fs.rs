//@ignore-target: windows # no libc
//@compile-flags: -Zmiri-disable-isolation
//@run-native

use std::ffi::{CStr, CString, OsString};
use std::fs::{self, File, canonicalize, create_dir, remove_dir, remove_file};
use std::io::{Error, ErrorKind, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::ptr;

#[path = "../../utils/mod.rs"]
mod utils;

#[path = "../../utils/libc.rs"]
mod libc_utils;

use libc_utils::errno_result;

fn main() {
    test_dup();
    test_dup_stdout_stderr();
    test_canonicalize_too_long();
    test_rename();
    test_ftruncate::<libc::off_t>(libc::ftruncate);
    #[cfg(target_os = "linux")]
    test_ftruncate::<libc::off64_t>(libc::ftruncate64);
    test_file_open_unix_allow_two_args();
    test_file_open_unix_needs_three_args();
    test_file_open_unix_extra_third_arg();
    test_file_open_dir();
    #[cfg(target_os = "linux")]
    test_o_tmpfile_flag();
    test_posix_mkstemp();
    test_posix_realpath_alloc();
    test_posix_realpath_noalloc();
    test_posix_realpath_errors();
    #[cfg(target_os = "linux")]
    test_posix_fadvise();
    #[cfg(not(target_os = "macos"))]
    test_posix_fallocate::<libc::off_t>(libc::posix_fallocate);
    #[cfg(target_os = "linux")]
    test_posix_fallocate::<libc::off64_t>(libc::posix_fallocate64);
    #[cfg(target_os = "linux")]
    test_sync_file_range();
    test_fstat();
    test_stat();
    test_lstat();
    test_isatty();
    test_read_and_uninit();
    test_nofollow_not_symlink();
    #[cfg(target_os = "macos")]
    test_ioctl();
    test_opendir_closedir();
    test_readdir();
    #[cfg(target_os = "linux")]
    test_statx_on_file_path();
    #[cfg(target_os = "linux")]
    test_statx_on_file_descriptor();
    #[cfg(target_os = "linux")]
    test_statx_empty_path_on_pipe();
    test_readv();
    test_readv_empty_bufs();
    #[cfg(not(target_os = "solaris"))]
    test_preadv();
    test_pread();
    test_writev();
    test_writev_empty_bufs();
    #[cfg(not(target_os = "solaris"))]
    test_pwritev();
    test_pwrite();
    test_linkat();
}

#[cfg(target_os = "linux")]
#[track_caller]
fn assert_statx_matches_metadata(stx: &libc::statx, meta: &fs::Metadata, expected_size: u64) {
    use std::os::unix::fs::MetadataExt;
    let mask = stx.stx_mask;

    // Guaranteed by the shim on any Linux target.
    assert!(mask & libc::STATX_SIZE != 0);
    assert_eq!(stx.stx_size, expected_size);
    assert!(mask & libc::STATX_TYPE != 0);
    assert_eq!((stx.stx_mode as libc::mode_t) & libc::S_IFMT, libc::S_IFREG);
    assert!(mask & libc::STATX_MODE != 0);
    assert_ne!((stx.stx_mode as libc::mode_t) & !libc::S_IFMT, 0);

    // Host-dependent enrichment: only assert when the mask says the field is real.
    if mask & libc::STATX_INO != 0 {
        assert_eq!(stx.stx_ino, meta.ino());
    }
    if mask & libc::STATX_NLINK != 0 {
        assert_eq!(stx.stx_nlink as u64, meta.nlink());
    }
    if mask & libc::STATX_UID != 0 {
        assert_eq!(stx.stx_uid, meta.uid());
    }
    if mask & libc::STATX_GID != 0 {
        assert_eq!(stx.stx_gid, meta.gid());
    }
    if mask & libc::STATX_BLOCKS != 0 {
        assert_eq!(stx.stx_blocks, meta.blocks());
    }

    // Do not assert stx_blksize and stx_dev_* : there are no mask bits for them.
}

#[cfg(target_os = "linux")]
fn test_statx_on_file_descriptor() {
    use std::mem::MaybeUninit;

    let bytes = b"hello";
    let path = utils::prepare_with_content("miri_test_libc_statx_fd.txt", bytes);
    let file = File::open(&path).unwrap();

    unsafe {
        let mut stx = MaybeUninit::<libc::statx>::zeroed();
        let ret = libc::statx(
            file.as_raw_fd(),
            c"".as_ptr(),
            libc::AT_EMPTY_PATH,
            libc::STATX_BASIC_STATS | libc::STATX_BTIME,
            stx.as_mut_ptr(),
        );
        assert_eq!(ret, 0, "statx failed: {}", std::io::Error::last_os_error());

        let stx = stx.assume_init();
        let meta = file.metadata().unwrap();
        assert_statx_matches_metadata(&stx, &meta, bytes.len() as u64);
    }

    drop(file);
    remove_file(&path).unwrap();
}

#[cfg(target_os = "linux")]
fn test_statx_on_file_path() {
    use std::mem::MaybeUninit;

    let bytes = b"hello";
    let path = utils::prepare_with_content("miri_test_libc_statx.txt", bytes);
    let c_path = CString::new(path.as_os_str().as_bytes()).expect("CString::new failed");

    unsafe {
        let mut stx = MaybeUninit::<libc::statx>::zeroed();
        let ret = libc::statx(
            libc::AT_FDCWD,
            c_path.as_ptr(),
            0,
            libc::STATX_BASIC_STATS | libc::STATX_BTIME,
            stx.as_mut_ptr(),
        );
        assert_eq!(ret, 0, "statx failed: {}", std::io::Error::last_os_error());

        let stx = stx.assume_init();
        let meta = fs::metadata(&path).unwrap();
        assert_statx_matches_metadata(&stx, &meta, bytes.len() as u64);
    }

    remove_file(&path).unwrap();
}

#[cfg(target_os = "linux")]
fn test_statx_empty_path_on_pipe() {
    use libc_utils::errno_check;

    unsafe {
        let mut fds = [0; 2];
        errno_check(libc::pipe(fds.as_mut_ptr()));

        let mut statx_buf = std::mem::MaybeUninit::<libc::statx>::zeroed();

        let ret = libc::statx(
            fds[0],
            c"".as_ptr(),
            libc::AT_EMPTY_PATH,
            libc::STATX_BASIC_STATS,
            statx_buf.as_mut_ptr(),
        );

        assert_eq!(
            ret,
            0,
            "statx on pipe with AT_EMPTY_PATH failed: {}",
            std::io::Error::last_os_error()
        );

        let statx_buf = statx_buf.assume_init();

        assert_ne!(statx_buf.stx_mask & libc::STATX_SIZE, 0);
        assert_eq!(statx_buf.stx_size, 0);
        assert_ne!(statx_buf.stx_mask & libc::STATX_TYPE, 0);
        assert_eq!((statx_buf.stx_mode as libc::mode_t) & libc::S_IFMT, libc::S_IFIFO);
        assert_ne!(statx_buf.stx_mask & libc::STATX_MODE, 0);
        assert_ne!((statx_buf.stx_mode as libc::mode_t) & !libc::S_IFMT, 0);

        if cfg!(miri) {
            // Synthetic metadata must not advertise host-only fields.
            assert_eq!(statx_buf.stx_mask & libc::STATX_INO, 0);
            assert_eq!(statx_buf.stx_mask & libc::STATX_NLINK, 0);
            assert_eq!(statx_buf.stx_mask & libc::STATX_UID, 0);
            assert_eq!(statx_buf.stx_mask & libc::STATX_GID, 0);
            assert_eq!(statx_buf.stx_mask & libc::STATX_BLOCKS, 0);
        }

        errno_check(libc::close(fds[0]));
        errno_check(libc::close(fds[1]));
    }
}

fn test_file_open_unix_allow_two_args() {
    let path = utils::prepare_with_content("test_file_open_unix_allow_two_args.txt", &[]);
    let name = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();

    let _fd = errno_result(unsafe { libc::open(name.as_ptr(), libc::O_RDONLY) }).unwrap();
}

fn test_file_open_unix_needs_three_args() {
    let path = utils::prepare_with_content("test_file_open_unix_needs_three_args.txt", &[]);
    let name = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();

    let _fd =
        errno_result(unsafe { libc::open(name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o666) })
            .unwrap();
}

fn test_file_open_unix_extra_third_arg() {
    let path = utils::prepare_with_content("test_file_open_unix_extra_third_arg.txt", &[]);
    let name = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();

    let _fd = errno_result(unsafe { libc::open(name.as_ptr(), libc::O_RDONLY, 42) }).unwrap();
}

fn test_file_open_dir() {
    let dir_path = utils::prepare_dir("miri_test_fs_dir");
    create_dir(&dir_path).unwrap();
    let dir_name = CString::new(dir_path.into_os_string().into_encoded_bytes()).unwrap();

    // Opening it for read-write fails. The error code differs between Unix and Windows hosts.
    let err = errno_result(unsafe { libc::open(dir_name.as_ptr(), libc::O_RDWR) }).unwrap_err();
    assert!(
        [libc::EISDIR, libc::EPERM].contains(&err.raw_os_error().unwrap()),
        "unexpected errno: {err}"
    );

    // Opening it for reading succeeds, but then reading fails.
    // FIXME: currently does not behave as expected on Windows hosts.
    // See <https://github.com/rust-lang/miri/issues/5084>.
    // let fd = errno_result(unsafe { libc::open(dir_name.as_ptr(), libc::O_RDONLY) }).unwrap();
    // let mut buf = [0u8; 4];
    // let err =
    //     errno_result(unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) }).unwrap_err();
    // assert_eq!(err.raw_os_error().unwrap(), libc::EISDIR, "unexpected errno: {err}");
    // libc_utils::errno_check(unsafe { libc::close(fd) });
}

fn test_dup_stdout_stderr() {
    let bytes = b"hello dup fd\n";
    unsafe {
        let new_stdout = libc::fcntl(1, libc::F_DUPFD, 0);
        let new_stderr = libc::fcntl(2, libc::F_DUPFD, 0);
        libc_utils::write_all(new_stdout, bytes).unwrap();
        libc_utils::write_all(new_stderr, bytes).unwrap();
    }
}

fn test_dup() {
    let bytes = b"dup and dup2";
    let path = utils::prepare_with_content("miri_test_libc_dup.txt", bytes);
    let name = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();

    unsafe {
        let fd = errno_result(libc::open(name.as_ptr(), libc::O_RDONLY)).unwrap();
        let new_fd = libc::dup(fd);
        let new_fd2 = libc::dup2(fd, 8);

        let mut first_buf = [0u8; 4];
        let first_len = libc::read(fd, first_buf.as_mut_ptr() as *mut libc::c_void, 4);
        assert!(first_len > 0);
        let first_len = first_len as usize;
        assert_eq!(first_buf[..first_len], bytes[..first_len]);
        let remaining_bytes = &bytes[first_len..];

        let mut second_buf = [0u8; 4];
        let second_len = libc::read(new_fd, second_buf.as_mut_ptr() as *mut libc::c_void, 4);
        assert!(second_len > 0);
        let second_len = second_len as usize;
        assert_eq!(second_buf[..second_len], remaining_bytes[..second_len]);
        let remaining_bytes = &remaining_bytes[second_len..];

        let mut third_buf = [0u8; 4];
        let third_len = libc::read(new_fd2, third_buf.as_mut_ptr() as *mut libc::c_void, 4);
        assert!(third_len > 0);
        let third_len = third_len as usize;
        assert_eq!(third_buf[..third_len], remaining_bytes[..third_len]);
    }
}

fn test_canonicalize_too_long() {
    // Make sure we get an error for long paths.
    let too_long = "x/".repeat(libc::PATH_MAX.try_into().unwrap());
    assert!(canonicalize(too_long).is_err());
}

fn test_rename() {
    let path1 = utils::prepare("miri_test_libc_fs_source.txt");
    let path2 = utils::prepare("miri_test_libc_fs_rename_destination.txt");

    let file = File::create(&path1).unwrap();
    drop(file);

    let c_path1 = CString::new(path1.as_os_str().as_bytes()).expect("CString::new failed");
    let c_path2 = CString::new(path2.as_os_str().as_bytes()).expect("CString::new failed");

    // Renaming should succeed
    unsafe { libc::rename(c_path1.as_ptr(), c_path2.as_ptr()) };
    // Check that old file path isn't present
    assert_eq!(ErrorKind::NotFound, path1.metadata().unwrap_err().kind());
    // Check that the file has moved successfully
    assert!(path2.metadata().unwrap().is_file());

    // Renaming a nonexistent file should fail
    let res = unsafe { libc::rename(c_path1.as_ptr(), c_path2.as_ptr()) };
    assert_eq!(res, -1);
    assert_eq!(Error::last_os_error().kind(), ErrorKind::NotFound);

    remove_file(&path2).unwrap();
}

fn test_ftruncate<T: From<i32>>(
    ftruncate: unsafe extern "C" fn(fd: libc::c_int, length: T) -> libc::c_int,
) {
    // libc::off_t is i32 in target i686-unknown-linux-gnu
    // https://docs.rs/libc/latest/i686-unknown-linux-gnu/libc/type.off_t.html

    let bytes = b"hello";
    let path = utils::prepare("miri_test_libc_fs_ftruncate.txt");
    let mut file = File::create(&path).unwrap();
    file.write_all(bytes).unwrap();
    file.sync_all().unwrap();
    assert_eq!(file.metadata().unwrap().len(), 5);

    let c_path = CString::new(path.as_os_str().as_bytes()).expect("CString::new failed");
    let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDWR) };

    // Truncate to a bigger size
    let mut res = unsafe { ftruncate(fd, T::from(10)) };
    assert_eq!(res, 0);
    assert_eq!(file.metadata().unwrap().len(), 10);

    // Write after truncate
    file.write(b"dup").unwrap();
    file.sync_all().unwrap();
    assert_eq!(file.metadata().unwrap().len(), 10);

    // Truncate to smaller size
    res = unsafe { ftruncate(fd, T::from(2)) };
    assert_eq!(res, 0);
    assert_eq!(file.metadata().unwrap().len(), 2);

    remove_file(&path).unwrap();
}

#[cfg(target_os = "linux")]
fn test_o_tmpfile_flag() {
    if !cfg!(miri) {
        return; // checks miri-specific behavior
    }

    use std::fs::{OpenOptions, create_dir};
    use std::os::unix::fs::OpenOptionsExt;
    let dir_path = utils::prepare_dir("miri_test_fs_dir");
    create_dir(&dir_path).unwrap();
    // test that the `O_TMPFILE` custom flag gracefully errors instead of stopping execution
    assert_eq!(
        OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_TMPFILE)
            .open(dir_path)
            .unwrap_err()
            .raw_os_error(),
        Some(libc::EOPNOTSUPP),
    );
}

fn test_posix_mkstemp() {
    use std::env;
    use std::ffi::OsStr;
    use std::os::unix::io::FromRawFd;
    use std::path::Path;

    // We want to test `mkstemp` on a relative name, so we cd to a tempdir and later cd back.
    let old_cwd = env::current_dir().unwrap();
    let dir_path = utils::prepare_dir("miri_test_libc_readdir");
    create_dir(&dir_path).expect("create_dir failed");
    env::set_current_dir(&dir_path).unwrap();

    let valid_template = "fooXXXXXX";
    // C needs to own this as `mkstemp(3)` says:
    // "Since it will be modified, `template` must not be a string constant, but
    // should be declared as a character array."
    // There seems to be no `as_mut_ptr` on `CString` so we need to use `into_raw`.
    let ptr = CString::new(valid_template).unwrap().into_raw();
    let fd = unsafe { libc::mkstemp(ptr) };
    assert!(fd >= 0, "mkstemp failed");
    // Take ownership back in Rust to not leak memory.
    let slice = unsafe { CString::from_raw(ptr) };
    let osstr = OsStr::from_bytes(slice.to_bytes());
    let path: &Path = osstr.as_ref();
    let name = path.to_string_lossy();
    assert!(name.ne("fooXXXXXX"));
    assert!(name.starts_with("foo"));
    assert_eq!(name.len(), 9);
    assert_eq!(
        name.chars().skip(3).filter(char::is_ascii_alphanumeric).collect::<Vec<char>>().len(),
        6
    );
    let file = unsafe { File::from_raw_fd(fd) };
    assert!(file.set_len(0).is_ok());
    // Cleanup. Also checks that the filename actually exists.
    drop(file);
    remove_file(path).unwrap();

    // Test invalid inputs. We skip this on native macOS since macOS apparently does
    // not bother to validate inputs.
    if !cfg!(all(not(miri), target_vendor = "apple")) {
        let invalid_templates = vec!["foo", "barXX", "XXXXXXbaz", "whatXXXXXXever", "X"];
        for t in invalid_templates {
            let ptr = CString::new(t).unwrap().into_raw();
            let fd = unsafe { libc::mkstemp(ptr) };
            let _ = unsafe { CString::from_raw(ptr) };
            // "On error, -1 is returned, and errno is set to
            // indicate the error"
            assert_eq!(fd, -1, "mkstemp succeeded on invalid template {t:?}");
            let e = std::io::Error::last_os_error();
            assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
        }
    }

    env::set_current_dir(old_cwd).unwrap();
}

/// Test allocating variant of `realpath`.
fn test_posix_realpath_alloc() {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};

    let buf;
    let path = utils::tmp().join("miri_test_libc_posix_realpath_alloc");
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

    let path = utils::tmp().join("miri_test_libc_posix_realpath_noalloc");
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

    // Test nonexistent path returns an error.
    let c_path = CString::new("./nothing_to_see_here").expect("CString::new failed");
    let r = unsafe { libc::realpath(c_path.as_ptr(), std::ptr::null_mut()) };
    assert!(r.is_null());
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::ENOENT));
    assert_eq!(e.kind(), ErrorKind::NotFound);
}

#[cfg(target_os = "linux")]
fn test_posix_fadvise() {
    use std::io::Write;

    let path = utils::tmp().join("miri_test_libc_posix_fadvise.txt");
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

#[cfg(not(target_os = "macos"))]
fn test_posix_fallocate<T: From<i32>>(
    posix_fallocate: unsafe extern "C" fn(fd: libc::c_int, offset: T, len: T) -> libc::c_int,
) {
    // libc::off_t is i32 in target i686-unknown-linux-gnu
    // https://docs.rs/libc/latest/i686-unknown-linux-gnu/libc/type.off_t.html

    let test_errors = || {
        // invalid fd
        let ret = unsafe { posix_fallocate(42, T::from(0), T::from(10)) };
        assert_eq!(ret, libc::EBADF);

        let path = utils::prepare("miri_test_libc_posix_fallocate_errors.txt");
        let file = File::create(&path).unwrap();

        // invalid offset
        let ret = unsafe { posix_fallocate(file.as_raw_fd(), T::from(-10), T::from(10)) };
        assert_eq!(ret, libc::EINVAL);

        // invalid len
        let ret = unsafe { posix_fallocate(file.as_raw_fd(), T::from(0), T::from(-10)) };
        assert_eq!(ret, libc::EINVAL);

        // fd not writable
        let c_path = CString::new(path.as_os_str().as_bytes()).expect("CString::new failed");
        let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY) };
        let ret = unsafe { posix_fallocate(fd, T::from(0), T::from(10)) };
        assert_eq!(ret, libc::EBADF);
    };

    let test = || {
        let bytes = b"hello";
        let path = utils::prepare("miri_test_libc_posix_fallocate.txt");
        let mut file = File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
        assert_eq!(file.metadata().unwrap().len(), 5);

        let c_path = CString::new(path.as_os_str().as_bytes()).expect("CString::new failed");
        let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDWR) };

        // Allocate to a bigger size from offset 0
        let mut res = unsafe { posix_fallocate(fd, T::from(0), T::from(10)) };
        assert_eq!(res, 0);
        assert_eq!(file.metadata().unwrap().len(), 10);

        // Write after allocation
        file.write(b"dup").unwrap();
        file.sync_all().unwrap();
        assert_eq!(file.metadata().unwrap().len(), 10);

        // Can't truncate to a smaller size with possix_fallocate
        res = unsafe { posix_fallocate(fd, T::from(0), T::from(3)) };
        assert_eq!(res, 0);
        assert_eq!(file.metadata().unwrap().len(), 10);

        // Allocate from offset
        res = unsafe { posix_fallocate(fd, T::from(7), T::from(7)) };
        assert_eq!(res, 0);
        assert_eq!(file.metadata().unwrap().len(), 14);

        remove_file(&path).unwrap();
    };

    test_errors();
    test();
}

#[cfg(target_os = "linux")]
fn test_sync_file_range() {
    use std::io::Write;

    let path = utils::tmp().join("miri_test_libc_sync_file_range.txt");
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

fn test_fstat() {
    use std::mem::MaybeUninit;
    use std::os::unix::io::AsRawFd;

    let path = utils::prepare_with_content("miri_test_libc_fstat.txt", b"hello");
    let file = File::open(&path).unwrap();
    let fd = file.as_raw_fd();

    let mut stat = MaybeUninit::<libc::stat>::uninit();
    let res = unsafe { libc::fstat(fd, stat.as_mut_ptr()) };
    assert_eq!(res, 0);
    let stat = unsafe { stat.assume_init_ref() };

    assert_eq!(stat.st_size, 5);
    assert_eq!(stat.st_mode & libc::S_IFMT, libc::S_IFREG);
    assert_ne!(stat.st_mode & !libc::S_IFMT, 0, "some permission should be set");

    // Check that all fields are initialized.
    check_stat_fields(stat);

    remove_file(&path).unwrap();
}

fn test_stat() {
    use std::mem::MaybeUninit;

    let path = utils::prepare_with_content("miri_test_libc_stat.txt", b"hello");
    let cpath = CString::new(path.as_os_str().as_bytes()).unwrap();

    let mut stat = MaybeUninit::<libc::stat>::uninit();
    let res = unsafe { libc::stat(cpath.as_ptr(), stat.as_mut_ptr()) };
    assert_eq!(res, 0);
    let stat = unsafe { stat.assume_init_ref() };

    assert_eq!(stat.st_size, 5);
    assert_eq!(stat.st_mode & libc::S_IFMT, libc::S_IFREG);
    assert_ne!(stat.st_mode & !libc::S_IFMT, 0, "some permission should be set");

    // Check that all fields are initialized.
    check_stat_fields(stat);

    remove_file(&path).unwrap();
}

fn test_lstat() {
    use std::mem::MaybeUninit;

    let path = utils::prepare_with_content("miri_test_libc_lstat.txt", b"hello");
    let symlink_path = utils::prepare("miri_test_libc_lstat_symlink.txt");

    std::os::unix::fs::symlink(&path, &symlink_path).unwrap();

    let cpath = CString::new(symlink_path.as_os_str().as_bytes()).unwrap();

    let mut stat = MaybeUninit::<libc::stat>::uninit();
    let res = unsafe { libc::lstat(cpath.as_ptr(), stat.as_mut_ptr()) };
    assert_eq!(res, 0);
    let stat = unsafe { stat.assume_init_ref() };

    assert_eq!(stat.st_mode & libc::S_IFMT, libc::S_IFLNK);
    assert_ne!(stat.st_mode & !libc::S_IFMT, 0, "some permission should be set");

    // Check that all fields are initialized.
    check_stat_fields(stat);

    remove_file(&symlink_path).unwrap();
    remove_file(&path).unwrap();
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
        let path = utils::tmp().join("notatty.txt");
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

fn test_read_and_uninit() {
    use std::mem::MaybeUninit;
    {
        // We test that libc::read initializes its buffer.
        let path = utils::prepare_with_content("pass-libc-read-and-uninit.txt", &[1u8, 2, 3]);
        let cpath = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();
        unsafe {
            let fd = libc::open(cpath.as_ptr(), libc::O_RDONLY);
            assert_ne!(fd, -1);
            let mut buf: MaybeUninit<u8> = std::mem::MaybeUninit::uninit();
            assert_eq!(libc::read(fd, buf.as_mut_ptr().cast::<std::ffi::c_void>(), 1), 1);
            let buf = buf.assume_init();
            assert_eq!(buf, 1);
            assert_eq!(libc::close(fd), 0);
            assert_eq!(libc::unlink(cpath.as_ptr()), 0);
        }
    }
    {
        // We test that if we requested to read 4 bytes, but actually read 3 bytes, then
        // 3 bytes (not 4) will be overwritten, and remaining byte will be left as-is.
        let data = [1u8, 2, 3];
        let path = utils::prepare_with_content("pass-libc-read-and-uninit-2.txt", &data);
        let cpath = CString::new(path.clone().into_os_string().into_encoded_bytes()).unwrap();
        unsafe {
            let fd = libc::open(cpath.as_ptr(), libc::O_RDONLY);
            assert_ne!(fd, -1);
            let mut buf = [42u8; 5];
            let res = libc::read(fd, buf.as_mut_ptr().cast::<std::ffi::c_void>(), 4);
            assert!(res > 0 && res < 4);
            for i in 0..buf.len() {
                assert_eq!(
                    buf[i],
                    if i < res as usize { data[i] } else { 42 },
                    "wrong result at pos {i}"
                );
            }
            assert_eq!(libc::close(fd), 0);
        }
        remove_file(&path).unwrap();
    }
}

fn test_nofollow_not_symlink() {
    let bytes = b"Hello, World!\n";
    let path = utils::prepare_with_content("test_nofollow_not_symlink.txt", bytes);
    let cpath = CString::new(path.as_os_str().as_bytes()).unwrap();
    let ret = unsafe { libc::open(cpath.as_ptr(), libc::O_NOFOLLOW | libc::O_CLOEXEC) };
    assert!(ret >= 0);
}

#[cfg(target_os = "macos")]
fn test_ioctl() {
    let path = utils::prepare_with_content("miri_test_libc_ioctl.txt", &[]);
    let name = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();

    unsafe {
        // 100 surely is an invalid FD.
        assert_eq!(libc::ioctl(100, libc::FIOCLEX), -1);
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap();
        assert_eq!(errno, libc::EBADF);

        let fd = libc::open(name.as_ptr(), libc::O_RDONLY);
        assert_eq!(libc::ioctl(fd, libc::FIOCLEX), 0);
    }
}

fn test_opendir_closedir() {
    // dir should exist
    let path = utils::prepare_dir("miri_test_libc_opendir_closedir");
    create_dir(&path).expect("create_dir failed");
    let cpath = CString::new(path.as_os_str().as_bytes()).expect("CString::new failed");
    let dir: *mut libc::DIR = unsafe { libc::opendir(cpath.as_ptr()) };
    assert!(!dir.is_null());
    assert_eq!(unsafe { libc::closedir(dir) }, 0);

    // dir should not exist
    remove_dir(&path).unwrap();
    let dir: *mut libc::DIR = unsafe { libc::opendir(cpath.as_ptr()) };
    assert!(dir.is_null());
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::ENOENT));
    assert_eq!(e.kind(), ErrorKind::NotFound);

    // open normal file as dir should fail
    let file_path = utils::prepare_with_content("test_not_a_dir.txt", b"hello");
    let cfile = CString::new(file_path.as_os_str().as_bytes()).expect("CString::new failed");
    let dir: *mut libc::DIR = unsafe { libc::opendir(cfile.as_ptr()) };
    assert!(dir.is_null());
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::ENOTDIR));
    assert_eq!(e.kind(), ErrorKind::NotADirectory);
    remove_file(&file_path).unwrap();
}

fn test_readdir() {
    use std::fs::{create_dir, remove_dir, write};

    let dir_path = utils::prepare_dir("miri_test_libc_readdir");
    create_dir(&dir_path).ok();

    // Create test files
    let file1 = dir_path.join("file1.txt");
    let file2 = dir_path.join("file2.txt");
    write(&file1, b"content1").unwrap();
    write(&file2, b"content2").unwrap();

    let c_path = CString::new(dir_path.as_os_str().as_bytes()).unwrap();

    unsafe {
        let dirp = libc::opendir(c_path.as_ptr());
        assert!(!dirp.is_null());
        let mut entries = Vec::new();
        loop {
            cfg_select! {
                target_os = "macos" => {
                    // On macos we only support readdir_r as that's what std uses there.
                    use std::mem::MaybeUninit;
                    use libc::dirent;
                    let mut entry: MaybeUninit<dirent> = MaybeUninit::uninit();
                    let mut result: *mut dirent = std::ptr::null_mut();
                    let ret = libc::readdir_r(dirp, entry.as_mut_ptr(), &mut result);
                    assert_eq!(ret, 0);
                    let entry_ptr = result;
                }
                _ => {
                    let entry_ptr = libc::readdir(dirp);
                }
            }
            if entry_ptr.is_null() {
                break;
            }
            let name_ptr = std::ptr::addr_of!((*entry_ptr).d_name) as *const libc::c_char;
            let name = CStr::from_ptr(name_ptr);
            let name_str = name.to_string_lossy();
            entries.push(name_str.into_owned());
        }
        assert_eq!(libc::closedir(dirp), 0);
        entries.sort();
        assert_eq!(&entries, &[".", "..", "file1.txt", "file2.txt"]);
    }

    remove_file(&file1).unwrap();
    remove_file(&file2).unwrap();
    remove_dir(&dir_path).unwrap();
}

/// Check that all common fields of a `stat` struct are initialized.
pub fn check_stat_fields(stat: &libc::stat) {
    let _st_nlink = stat.st_nlink;
    let _st_blksize = stat.st_blksize;
    let _st_blocks = stat.st_blocks;
    let _st_ino = stat.st_ino;
    let _st_dev = stat.st_dev;
    let _st_uid = stat.st_uid;
    let _st_gid = stat.st_gid;
    let _st_rdev = stat.st_rdev;
    let _st_atime = stat.st_atime;
    let _st_mtime = stat.st_mtime;
    let _st_ctime = stat.st_ctime;
    let _st_atime_nsec = stat.st_atime_nsec;
    let _st_mtime_nsec = stat.st_mtime_nsec;
    let _st_ctime_nsec = stat.st_ctime_nsec;
}

/// Test vectored reads with multiple buffers.
fn test_readv() {
    let file_contents = [1u8, 2, 3, 4, 5, 6];
    let path = utils::prepare_with_content("pass-libc-readv.txt", &file_contents);
    let cpath = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY) };
    assert_ne!(fd, -1);

    let mut buffer = [0u8; 4];
    let (buffer1, buffer2) = buffer.split_at_mut(2);

    let iov = [
        libc::iovec { iov_base: ptr::null_mut::<libc::c_void>(), iov_len: 0 as libc::size_t },
        libc::iovec {
            iov_base: buffer1.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: buffer1.len() as libc::size_t,
        },
        libc::iovec {
            iov_base: buffer2.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: buffer2.len() as libc::size_t,
        },
    ];

    let bytes_read = unsafe {
        errno_result(libc::readv(fd, iov.as_ptr(), iov.len() as libc::c_int)).unwrap() as usize
    };

    // The vectored read should read at least one byte.
    assert!(bytes_read > 0);
    assert_eq!(&buffer[0..bytes_read], &file_contents[0..bytes_read]);
}

/// Test that vectored reads without any buffers return zero.
fn test_readv_empty_bufs() {
    if cfg!(all(not(miri), target_vendor = "apple")) {
        // native macOS returns an error here :shrug:
        return;
    }

    let path = utils::prepare_with_content("pass-libc-readv-empty-bufs.txt", &[1u8, 2, 3]);
    let cpath = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY) };
    assert_ne!(fd, -1);
    unsafe { assert_eq!(errno_result(libc::readv(fd, ptr::null::<libc::iovec>(), 0)).unwrap(), 0) };
}

/// Test vectored reads with multiple buffers and a byte offset.
///
/// **Note**: We skip this test on Solaris targets because Solaris
/// doesn't have `preadv`.
#[cfg(not(target_os = "solaris"))]
fn test_preadv() {
    let file_contents = [1u8, 2, 3, 4, 5, 6];
    let path = utils::prepare_with_content("pass-libc-preadv.txt", &file_contents);
    let cpath = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY) };
    assert_ne!(fd, -1);

    let mut buffer = [0u8; 4];
    let (buffer1, buffer2) = buffer.split_at_mut(2);

    let iov = [
        libc::iovec { iov_base: ptr::null_mut::<libc::c_void>(), iov_len: 0 as libc::size_t },
        libc::iovec {
            iov_base: buffer1.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: buffer1.len() as libc::size_t,
        },
        libc::iovec {
            iov_base: buffer2.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: buffer2.len() as libc::size_t,
        },
    ];

    // Read with a 2 byte offset.
    const OFFSET: usize = 2;
    let bytes_read = unsafe {
        errno_result(libc::preadv(
            fd,
            iov.as_ptr(),
            iov.len() as libc::c_int,
            OFFSET as libc::off_t,
        ))
        .unwrap() as usize
    };

    // The vectored read should read at least one byte.
    assert!(bytes_read > 0);
    // The vectored read should start at the provided byte offset.
    assert_eq!(&buffer[0..bytes_read], &file_contents[OFFSET..(bytes_read + OFFSET)]);
}

/// Test reading with an offset.
fn test_pread() {
    let file_contents = [1u8, 2, 3, 4, 5, 6];
    let path = utils::prepare_with_content("pass-libc-pread.txt", &file_contents);
    let cpath = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY) };
    assert_ne!(fd, -1);

    let mut buffer = [0u8; 2];

    // Read with a 2 byte offset.
    const OFFSET: usize = 2;
    let bytes_read = unsafe {
        errno_result(libc::pread(
            fd,
            buffer.as_mut_ptr().cast(),
            buffer.len() as libc::size_t,
            OFFSET as libc::off_t,
        ))
        .unwrap() as usize
    };

    // We should read at least one byte.
    assert!(bytes_read > 0);
    // The read should start at the provided byte offset.
    assert_eq!(&buffer[0..bytes_read], &file_contents[OFFSET..(bytes_read + OFFSET)]);
}

/// Test vectored writes with multiple buffers.
fn test_writev() {
    let path = utils::prepare_with_content("pass-libc-writev.txt", &[]);
    let cpath = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_WRONLY) };
    assert_ne!(fd, -1);

    let mut write_buffer = [1u8, 2, 3, 4, 5, 6];
    let (buffer1, buffer2) = write_buffer.split_at_mut(3);

    let iov = [
        libc::iovec { iov_base: ptr::null_mut::<libc::c_void>(), iov_len: 0 as libc::size_t },
        libc::iovec {
            iov_base: buffer1.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: buffer1.len() as libc::size_t,
        },
        libc::iovec {
            iov_base: buffer2.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: buffer2.len() as libc::size_t,
        },
    ];

    let bytes_written = unsafe {
        errno_result(libc::writev(fd, iov.as_ptr(), iov.len() as libc::c_int)).unwrap() as usize
    };
    // The vectored write should write at least one byte.
    assert!(bytes_written > 0);

    // Open the FD again in readonly mode and with an unadvanced pointer.
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY) };
    assert_ne!(fd, -1);

    let mut read_buffer = [0u8; 16];
    unsafe {
        libc_utils::read_exact_generic(
            read_buffer.as_mut_ptr().cast(),
            bytes_written as libc::size_t,
            libc_utils::Retry::NoRetry,
            |buf, count| libc::read(fd, buf, count),
        )
        .unwrap()
    };

    assert_eq!(&write_buffer[0..bytes_written], &read_buffer[0..bytes_written]);
}

/// Test that vectored writes without any buffers return zero.
fn test_writev_empty_bufs() {
    if cfg!(all(not(miri), target_vendor = "apple")) {
        // native macOS returns an error here :shrug:
        return;
    }

    let path = utils::prepare_with_content("pass-libc-writev-empty-bufs.txt", &[1u8, 2, 3]);
    let cpath = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_WRONLY) };
    assert_ne!(fd, -1);
    unsafe {
        assert_eq!(errno_result(libc::writev(fd, ptr::null::<libc::iovec>(), 0)).unwrap(), 0)
    };
}

/// Test vectored writes with multiple buffers and a byte offset.
///
/// **Note**: We skip this test on Solaris targets because Solaris
/// doesn't have `pwritev`.
#[cfg(not(target_os = "solaris"))]
fn test_pwritev() {
    let path = utils::prepare_with_content("pass-libc-pwritev.txt", &[]);
    let cpath = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_WRONLY) };
    assert_ne!(fd, -1);

    let mut write_buffer = [1u8, 2, 3, 4, 5, 6];
    let (buffer1, buffer2) = write_buffer.split_at_mut(3);

    let iov = [
        libc::iovec { iov_base: ptr::null_mut::<libc::c_void>(), iov_len: 0 as libc::size_t },
        libc::iovec {
            iov_base: buffer1.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: buffer1.len() as libc::size_t,
        },
        libc::iovec {
            iov_base: buffer2.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: buffer2.len() as libc::size_t,
        },
    ];

    // Write with a 2 byte offset.
    const OFFSET: usize = 2;
    let bytes_written = unsafe {
        errno_result(libc::pwritev(
            fd,
            iov.as_ptr(),
            iov.len() as libc::c_int,
            OFFSET as libc::off_t,
        ))
        .unwrap() as usize
    };
    // The vectored write should write at least one byte.
    assert!(bytes_written > 0);

    // Open the FD again in readonly mode and with an unadvanced pointer.
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY) };
    assert_ne!(fd, -1);

    let mut read_buffer = [0u8; 16];
    // Read offset + bytes written.
    unsafe {
        libc_utils::read_exact_generic(
            read_buffer.as_mut_ptr().cast(),
            (bytes_written + OFFSET) as libc::size_t,
            libc_utils::Retry::NoRetry,
            |buf, count| libc::read(fd, buf, count),
        )
        .unwrap()
    };

    // The vectored write should start at the provided byte offset.
    assert_eq!(&write_buffer[0..bytes_written], &read_buffer[OFFSET..(bytes_written + OFFSET)]);
}

/// Test writing with an offset.
fn test_pwrite() {
    let path = utils::prepare_with_content("pass-libc-pwritev.txt", &[]);
    let cpath = CString::new(path.into_os_string().into_encoded_bytes()).unwrap();
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_WRONLY) };
    assert_ne!(fd, -1);

    let write_buffer = [1u8, 2, 3, 4, 5, 6];

    // Write with a 2 byte offset.
    const OFFSET: usize = 2;
    let bytes_written = unsafe {
        errno_result(libc::pwrite(
            fd,
            write_buffer.as_ptr().cast(),
            write_buffer.len() as libc::size_t,
            OFFSET as libc::off_t,
        ))
        .unwrap() as usize
    };
    // We should write at least one byte.
    assert!(bytes_written > 0);

    // Open the FD again in readonly mode and with an unadvanced pointer.
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY) };
    assert_ne!(fd, -1);

    let mut read_buffer = [0u8; 16];
    // Read offset + bytes written.
    unsafe {
        libc_utils::read_exact_generic(
            read_buffer.as_mut_ptr().cast(),
            (bytes_written + OFFSET) as libc::size_t,
            libc_utils::Retry::NoRetry,
            |buf, count| libc::read(fd, buf, count),
        )
        .unwrap()
    };

    // The write should start at the provided byte offset.
    assert_eq!(&write_buffer[0..bytes_written], &read_buffer[OFFSET..(bytes_written + OFFSET)]);
}

fn test_linkat() {
    let source = utils::prepare_with_content("miri_test_libc_linkat_source.txt", b"hello");
    let link = utils::prepare("miri_test_libc_linkat_link.txt");

    let c_source = CString::new(source.as_os_str().as_bytes()).expect("CString::new failed");
    let c_link = CString::new(link.as_os_str().as_bytes()).expect("CString::new failed");

    // Call linkat
    unsafe {
        libc_utils::errno_check(libc::linkat(
            libc::AT_FDCWD,
            c_source.as_ptr(),
            libc::AT_FDCWD,
            c_link.as_ptr(),
            0,
        ));
    }

    // Verify that the hard link works:
    // Modifications to one are visible through the other.
    fs::write(&source, b"hello world").unwrap();
    let contents = fs::read(&link).unwrap();
    assert_eq!(contents, b"hello world");

    // Cleanup
    remove_file(&source).unwrap();
    remove_file(&link).unwrap();
}
