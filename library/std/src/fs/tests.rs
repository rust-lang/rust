use rand::RngCore;
use std::fs::{self, File, FileTimes, OpenOptions, TryLockError};
use std::io::prelude::*;
use std::io::{BorrowedBuf, ErrorKind, SeekFrom};
use std::mem::MaybeUninit;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

#[cfg(unix)]
use std::os::unix::fs::{symlink as symlink_file, symlink as symlink_dir, PermissionsExt};
#[cfg(windows)]
use std::os::windows::fs::{
    junction_point, symlink_dir, symlink_file, OpenOptionsExt, FileExt as WindowsFileExt,
};

use crate::assert_matches::assert_matches;
use crate::test_helpers::{tmpdir, TempDir};

macro_rules! check {
    ($e:expr) => {
        match $e {
            Ok(t) => t,
            Err(e) => panic!("{} failed with: {e}", stringify!($e)),
        }
    };
}

#[cfg(windows)]
macro_rules! error {
    ($e:expr, $s:expr) => {
        match $e {
            Ok(_) => panic!("Unexpected success. Should've been: {:?}", $s),
            Err(ref err) => {
                assert!(
                    err.raw_os_error() == Some($s),
                    "`{}` did not have a code of `{}`",
                    err,
                    $s
                )
            }
        }
    };
}

#[cfg(unix)]
macro_rules! error {
    ($e:expr, $s:expr) => {
        error_contains!($e, $s)
    };
}

macro_rules! error_contains {
    ($e:expr, $s:expr) => {
        match $e {
            Ok(_) => panic!("Unexpected success. Should've been: {:?}", $s),
            Err(ref err) => {
                assert!(
                    err.to_string().contains($s),
                    "`{}` did not contain `{}`",
                    err,
                    $s
                )
            }
        }
    };
}

pub fn got_symlink_permission(tmpdir: &TempDir) -> bool {
    if !cfg!(windows) || std::env::var_os("CI").is_some() {
        return true;
    }
    let link = tmpdir.join("some_hopefully_unique_link_name");
    match symlink_file(r"nonexisting_target", link) {
        Err(ref err) if err.raw_os_error() == Some(1314) => false,
        Ok(_) | Err(_) => true,
    }
}

#[test]
fn file_test_io_smoke_test() {
    let message = "it's alright. have a good time";
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test.txt");
    {
        let mut write_stream = check!(File::create(&filename));
        check!(write_stream.write(message.as_bytes()));
    }
    {
        let mut read_stream = check!(File::open(&filename));
        let mut read_buf = [0u8; 1028];
        let n = check!(read_stream.read(&mut read_buf));
        assert_eq!(&read_buf[..n], message.as_bytes());
    }
    check!(fs::remove_file(&filename));
}

#[test]
fn invalid_path_raises() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_that_does_not_exist.txt");
    let result = File::open(&filename);
    #[cfg(all(unix, not(target_os = "vxworks")))]
    error!(result, "No such file or directory");
    #[cfg(target_os = "vxworks")]
    error!(result, "no such file or directory");
    #[cfg(windows)]
    error!(result, 2);
}

#[test]
fn file_test_iounlinking_invalid_path_should_raise_condition() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_another_file_that_does_not_exist.txt");
    let result = fs::remove_file(&filename);
    #[cfg(all(unix, not(target_os = "vxworks")))]
    error!(result, "No such file or directory");
    #[cfg(target_os = "vxworks")]
    error!(result, "no such file or directory");
    #[cfg(windows)]
    error!(result, 2);
}

#[test]
fn file_test_io_non_positional_read() {
    let message: &str = "ten-four";
    let mut read_mem = [0u8; 8];
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_positional.txt");
    {
        let mut rw_stream = check!(File::create(&filename));
        check!(rw_stream.write(message.as_bytes()));
    }
    {
        let mut read_stream = check!(File::open(&filename));
        check!(read_stream.read(&mut read_mem[0..4]));
        check!(read_stream.read(&mut read_mem[4..8]));
    }
    check!(fs::remove_file(&filename));
    let read_str = std::str::from_utf8(&read_mem).unwrap();
    assert_eq!(read_str, message);
}

#[test]
fn file_test_io_seek_and_tell_smoke_test() {
    let message = "ten-four";
    let mut read_mem = [0u8; 256];
    let set_cursor = 4u64;
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_seeking.txt");
    {
        let mut rw_stream = check!(File::create(&filename));
        check!(rw_stream.write(message.as_bytes()));
    }
    {
        let mut read_stream = check!(File::open(&filename));
        check!(read_stream.seek(SeekFrom::Start(set_cursor)));
        let pre = check!(read_stream.stream_position());
        check!(read_stream.read(&mut read_mem));
        let post = check!(read_stream.stream_position());
        assert_eq!(pre, set_cursor);
        assert_eq!(post, message.len() as u64);
    }
    check!(fs::remove_file(&filename));
    assert_eq!(std::str::from_utf8(&read_mem[..4]).unwrap(), &message[4..]);
}

#[test]
fn file_test_io_seek_and_write() {
    let initial_msg = "food-is-yummy";
    let overwrite_msg = "-the-bar!!";
    let final_msg = "foo-the-bar!!";
    let seek_idx = 3u64;
    let mut read_mem = [0u8; 13];
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_seek_and_write.txt");
    {
        let mut rw_stream = check!(File::create(&filename));
        check!(rw_stream.write(initial_msg.as_bytes()));
        check!(rw_stream.seek(SeekFrom::Start(seek_idx)));
        check!(rw_stream.write(overwrite_msg.as_bytes()));
    }
    {
        let mut read_stream = check!(File::open(&filename));
        check!(read_stream.read(&mut read_mem));
    }
    check!(fs::remove_file(&filename));
    assert_eq!(std::str::from_utf8(&read_mem).unwrap(), final_msg);
}

#[test]
#[cfg_attr(
    not(any(
        windows,
        target_os = "aix",
        target_os = "cygwin",
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "illumos",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "solaris",
        target_vendor = "apple",
    )),
    should_panic
)]
fn file_lock_multiple_shared() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_lock_multiple_shared_test.txt");
    let f1 = check!(File::create(&filename));
    let f2 = check!(OpenOptions::new().write(true).open(&filename));
    check!(f1.lock_shared());
    check!(f2.lock_shared());
    check!(f1.unlock());
    check!(f2.unlock());
    check!(f1.try_lock_shared());
    check!(f2.try_lock_shared());
}

#[test]
#[cfg_attr(
    not(any(
        windows,
        target_os = "aix",
        target_os = "cygwin",
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "illumos",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "solaris",
        target_vendor = "apple",
    )),
    should_panic
)]
fn file_lock_blocking() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_lock_blocking_test.txt");
    let f1 = check!(File::create(&filename));
    let f2 = check!(OpenOptions::new().write(true).open(&filename));
    check!(f1.lock_shared());
    assert_matches!(f2.try_lock(), Err(TryLockError::WouldBlock));
    check!(f1.unlock());
    check!(f1.lock());
    assert_matches!(f2.try_lock_shared(), Err(TryLockError::WouldBlock));
}

#[test]
#[cfg_attr(
    not(any(
        windows,
        target_os = "aix",
        target_os = "cygwin",
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "illumos",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "solaris",
        target_vendor = "apple",
    )),
    should_panic
)]
fn file_lock_drop() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_lock_dup_test.txt");
    let f1 = check!(File::create(&filename));
    let f2 = check!(OpenOptions::new().write(true).open(&filename));
    check!(f1.lock_shared());
    assert_matches!(f2.try_lock(), Err(TryLockError::WouldBlock));
    drop(f1);
    check!(f2.try_lock());
}

#[test]
#[cfg_attr(
    not(any(
        windows,
        target_os = "aix",
        target_os = "cygwin",
        target_os = "freebsd",
        target_os = "fuchsia",
        target_os = "illumos",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "solaris",
        target_vendor = "apple",
    )),
    should_panic
)]
fn file_lock_dup() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_lock_dup_test.txt");
    let f1 = check!(File::create(&filename));
    let f2 = check!(OpenOptions::new().write(true).open(&filename));
    check!(f1.lock_shared());
    assert_matches!(f2.try_lock(), Err(TryLockError::WouldBlock));
    let cloned = check!(f1.try_clone());
    drop(f1);
    assert_matches!(f2.try_lock(), Err(TryLockError::WouldBlock));
    drop(cloned);
}

#[test]
#[cfg(windows)]
fn file_lock_double_unlock() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_lock_double_unlock_test.txt");
    let f1 = check!(File::create(&filename));
    let f2 = check!(OpenOptions::new().write(true).open(&filename));
    check!(f1.lock());
    check!(f1.lock_shared());
    assert_matches!(f2.try_lock(), Err(TryLockError::WouldBlock));
    check!(f1.unlock());
    check!(f2.try_lock());
}

#[test]
#[cfg(windows)]
fn file_lock_blocking_async() {
    use std::thread::{sleep, spawn};
    const FILE_FLAG_OVERLAPPED: u32 = 0x40000000;
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_lock_blocking_async.txt");
    let f1 = check!(File::create(&filename));
    let f2 = check!(OpenOptions::new().custom_flags(FILE_FLAG_OVERLAPPED).write(true).open(&filename));
    check!(f1.lock());
    let t = spawn(move || check!(f2.lock()));
    sleep(Duration::from_secs(1));
    assert!(!t.is_finished());
    check!(f1.unlock());
    t.join().unwrap();

    let f2 = check!(OpenOptions::new().custom_flags(FILE_FLAG_OVERLAPPED).write(true).open(&filename));
    check!(f1.lock());
    let t = spawn(move || check!(f2.lock_shared()));
    sleep(Duration::from_secs(1));
    assert!(!t.is_finished());
    check!(f1.unlock());
    t.join().unwrap();
}

#[test]
#[cfg(windows)]
fn file_try_lock_async() {
    const FILE_FLAG_OVERLAPPED: u32 = 0x40000000;
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_try_lock_async.txt");
    let f1 = check!(File::create(&filename));
    let f2 = check!(OpenOptions::new().custom_flags(FILE_FLAG_OVERLAPPED).write(true).open(&filename));
    check!(f1.lock_shared());
    assert_matches!(f2.try_lock(), Err(TryLockError::WouldBlock));
    check!(f1.unlock());
    check!(f1.lock());
    assert_matches!(f2.try_lock(), Err(TryLockError::WouldBlock));
    assert_matches!(f2.try_lock_shared(), Err(TryLockError::WouldBlock));
}

#[test]
fn file_test_io_seek_shakedown() {
    let initial_msg = "qwer-asdf-zxcv";
    let mut read_mem = [0u8; 256];
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_seek_shakedown.txt");
    {
        let mut rw_stream = check!(File::create(&filename));
        check!(rw_stream.write(initial_msg.as_bytes()));
    }
    {
        let mut read_stream = check!(File::open(&filename));
        check!(read_stream.seek(SeekFrom::End(-4)));
        check!(read_stream.read(&mut read_mem));
        assert_eq!(std::str::from_utf8(&read_mem[..4]).unwrap(), "zxcv");
        check!(read_stream.seek(SeekFrom::Current(-9)));
        check!(read_stream.read(&mut read_mem));
        assert_eq!(std::str::from_utf8(&read_mem[..4]).unwrap(), "asdf");
        check!(read_stream.seek(SeekFrom::Start(0)));
        check!(read_stream.read(&mut read_mem));
        assert_eq!(std::str::from_utf8(&read_mem[..4]).unwrap(), "qwer");
    }
    check!(fs::remove_file(&filename));
}

#[test]
fn file_test_io_eof() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_eof.txt");
    let mut buf = [0u8; 256];
    {
        let oo = OpenOptions::new().create_new(true).write(true).read(true);
        let mut rw = check!(oo.open(&filename));
        assert_eq!(check!(rw.read(&mut buf)), 0);
    }
    check!(fs::remove_file(&filename));
}

#[test]
#[cfg(unix)]
fn file_test_io_read_write_at() {
    use std::os::unix::fs::FileExt;
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_read_write_at.txt");
    let mut buf = [0u8; 256];
    let write1 = b"asdf";
    let write2 = b"qwer-";
    let write3 = b"-zxcv";
    let content = b"qwer-asdf-zxcv";
    {
        let mut rw = check!(OpenOptions::new().create_new(true).write(true).read(true).open(&filename));
        assert_eq!(check!(rw.write_at(write1, 5)), write1.len());
        assert_eq!(check!(rw.write(write2)), write2.len());
        assert_eq!(check!(rw.write_at(write3, 9)), write3.len());
    }
    {
        let mut read = check!(File::open(&filename));
        assert_eq!(check!(read.read_at(&mut buf, 0)), content.len());
        assert_eq!(&buf[..content.len()], content);
    }
    check!(fs::remove_file(&filename));
}

#[test]
#[cfg(unix)]
fn test_read_buf_at() {
    use std::os::unix::fs::FileExt;
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_read_buf_at.txt");
    {
        let mut file = check!(OpenOptions::new().create_new(true).write(true).open(&filename));
        check!(file.write_all(b"0123456789"));
    }
    {
        let mut file = check!(File::open(&filename));
        let mut buf = [MaybeUninit::<u8>::uninit(); 5];
        let mut buf = BorrowedBuf::from(buf.as_mut_slice());
        while buf.unfilled().capacity() > 0 {
            let len = buf.len();
            check!(file.read_buf_at(buf.unfilled(), 2 + len as u64));
        }
        assert_eq!(buf.filled(), b"23456");
    }
    check!(fs::remove_file(&filename));
}

#[test]
#[cfg(unix)]
fn test_read_buf_exact_at() {
    use std::os::unix::fs::FileExt;
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_read_buf_exact_at.txt");
    {
        let mut file = check!(OpenOptions::new().create_new(true).write(true).open(&filename));
        check!(file.write_all(b"0123456789"));
    }
    {
        let mut file = check!(File::open(&filename));
        let mut buf = [MaybeUninit::<u8>::uninit(); 5];
        let mut buf = BorrowedBuf::from(buf.as_mut_slice());
        check!(file.read_buf_exact_at(buf.unfilled(), 2));
        assert_eq!(buf.filled(), b"23456");
    }
    check!(fs::remove_file(&filename));
}

#[test]
#[cfg(unix)]
fn set_get_unix_permissions() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("set_get_unix_permissions");
    check!(fs::create_dir(&filename));
    check!(fs::set_permissions(&filename, fs::Permissions::from_mode(0)));
    let meta = check!(fs::metadata(&filename));
    assert_eq!(meta.permissions().mode() & 0o7777, 0);
    check!(fs::set_permissions(&filename, fs::Permissions::from_mode(0o1777)));
    let meta = check!(fs::metadata(&filename));
    assert_eq!(meta.permissions().mode() & 0o7777, 0o1777);
}

#[test]
#[cfg(windows)]
fn file_test_io_seek_read_write() {
    use std::os::windows::fs::FileExt;
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_seek_read_write.txt");
    let mut buf = [0u8; 256];
    let write1 = b"asdf";
    let write2 = b"qwer-";
    let write3 = b"-zxcv";
    let content = b"qwer-asdf-zxcv";
    {
        let mut rw = check!(OpenOptions::new().create_new(true).write(true).read(true).open(&filename));
        check!(rw.seek_write(write1, 5));
        check!(rw.seek(SeekFrom::Start(0)));
        check!(rw.write(write2));
        check!(rw.seek_write(write3, 9));
    }
    {
        let mut read = check!(File::open(&filename));
        assert_eq!(check!(read.seek_read(&mut buf, 0)), content.len());
        assert_eq!(&buf[..content.len()], content);
    }
    check!(fs::remove_file(&filename));
}

#[test]
#[cfg(windows)]
fn test_seek_read_buf() {
    use std::os::windows::fs::FileExt;
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_seek_read_buf.txt");
    {
        let mut file = check!(OpenOptions::new().create_new(true).write(true).open(&filename));
        check!(file.write_all(b"0123456789"));
    }
    {
        let mut file = check!(File::open(&filename));
        let mut buf = [MaybeUninit::<u8>::uninit(); 1];
        let mut buf = BorrowedBuf::from(buf.as_mut_slice());
        check!(file.seek_read_buf(buf.unfilled(), 8));
        assert_eq!(buf.filled(), b"8");
    }
    check!(fs::remove_file(&filename));
}

#[test]
fn file_test_read_buf() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("test");
    check!(fs::write(&filename, &[1, 2, 3, 4]));
    let mut buf = [MaybeUninit::<u8>::uninit(); 128];
    let mut buf = BorrowedBuf::from(buf.as_mut_slice());
    let mut file = check!(File::open(&filename));
    check!(file.read_buf(buf.unfilled()));
    assert_eq!(buf.filled(), &[1, 2, 3, 4]);
    check!(fs::remove_file(&filename));
}

#[test]
fn file_test_stat_is_correct_on_is_file() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_stat_correct_on_is_file.txt");
    let mut f = check!(OpenOptions::new().read(true).write(true).create(true).open(&filename));
    f.write_all(b"hw").unwrap();
    assert!(f.metadata().unwrap().is_file());
    assert!(filename.metadata().unwrap().is_file());
    check!(fs::remove_file(&filename));
}

#[test]
fn file_test_stat_is_correct_on_is_dir() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_stat_correct_on_is_dir");
    check!(fs::create_dir(&filename));
    assert!(filename.metadata().unwrap().is_dir());
    check!(fs::remove_dir(&filename));
}

#[test]
fn file_test_fileinfo_false_when_checking_is_file_on_a_directory() {
    let tmpdir = tmpdir();
    let dir = tmpdir.join("fileinfo_false_on_dir");
    check!(fs::create_dir(&dir));
    assert!(!dir.is_file());
    check!(fs::remove_dir(&dir));
}

#[test]
fn file_test_fileinfo_check_exists_before_and_after_file_creation() {
    let tmpdir = tmpdir();
    let file = tmpdir.join("fileinfo_check_exists_b_and_a.txt");
    check!(File::create(&file).and_then(|mut f| f.write(b"foo")));
    assert!(file.exists());
    check!(fs::remove_file(&file));
    assert!(!file.exists());
}

#[test]
fn file_test_directoryinfo_check_exists_before_and_after_mkdir() {
    let tmpdir = tmpdir();
    let dir = tmpdir.join("before_and_after_dir");
    assert!(!dir.exists());
    check!(fs::create_dir(&dir));
    assert!(dir.exists());
    assert!(dir.is_dir());
    check!(fs::remove_dir(&dir));
    assert!(!dir.exists());
}

#[test]
fn file_test_directoryinfo_readdir() {
    let tmpdir = tmpdir();
    let dir = tmpdir.join("di_readdir");
    check!(fs::create_dir(&dir));
    for n in 0..3 {
        let f = dir.join(format!("{n}.txt"));
        check!(File::create(&f).and_then(|mut w| w.write_all(format!("foo{n}").as_bytes())));
    }
    for entry in check!(fs::read_dir(&dir)) {
        let entry = check!(entry);
        let path = entry.path();
        let mut file = check!(File::open(&path));
        let mut s = String::new();
        check!(file.read_to_string(&mut s));
        assert!(s.starts_with("foo"));
    }
    check!(fs::remove_dir_all(&dir));
}

#[test]
fn file_create_new_already_exists_error() {
    let tmpdir = tmpdir();
    let file = tmpdir.join("file_create_new_error_exists");
    check!(File::create(&file));
    let e = OpenOptions::new().write(true).create_new(true).open(&file).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::AlreadyExists);
}

#[test]
fn mkdir_path_already_exists_error() {
    let tmpdir = tmpdir();
    let dir = tmpdir.join("mkdir_error_twice");
    check!(fs::create_dir(&dir));
    let e = fs::create_dir(&dir).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::AlreadyExists);
}

#[test]
fn recursive_mkdir() {
    let tmpdir = tmpdir();
    let dir = tmpdir.join("d1/d2");
    check!(fs::create_dir_all(&dir));
    assert!(dir.is_dir());
}

#[test]
fn recursive_mkdir_failure() {
    let tmpdir = tmpdir();
    let dir = tmpdir.join("d1");
    let file = dir.join("f1");
    check!(fs::create_dir_all(&dir));
    check!(File::create(&file));
    assert!(fs::create_dir_all(&file).is_err());
}

#[test]
fn concurrent_recursive_mkdir() {
    for _ in 0..100 {
        let dir = tmpdir();
        let mut path = dir.join("a");
        for _ in 0..40 {
            path = path.join("a");
        }
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let p = path.clone();
                thread::spawn(move || check!(fs::create_dir_all(&p)))
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }
}

#[test]
fn recursive_mkdir_slash() {
    check!(fs::create_dir_all(Path::new("/")));
}

#[test]
fn recursive_mkdir_dot() {
    check!(fs::create_dir_all(Path::new(".")));
}

#[test]
fn recursive_mkdir_empty() {
    check!(fs::create_dir_all(Path::new("")));
}

#[test]
#[cfg_attr(
    all(windows, target_arch = "aarch64"),
    ignore = "SymLinks not enabled on Arm64 Windows runners"
)]
fn recursive_rmdir() {
    let tmpdir = tmpdir();
    let d1 = tmpdir.join("d1");
    let dt = d1.join("t");
    let dtt = dt.join("t");
    let d2 = tmpdir.join("d2");
    let canary = d2.join("do_not_delete");
    check!(fs::create_dir_all(&dtt));
    check!(fs::create_dir_all(&d2));
    check!(File::create(&canary));
    check!(junction_point(&d2, &dt.join("d2")));
    let _ = symlink_file(&canary, &d1.join("canary"));
    check!(fs::remove_dir_all(&d1));
    assert!(!d1.is_dir());
    assert!(canary.exists());
}

#[test]
#[cfg_attr(
    all(windows, target_arch = "aarch64"),
    ignore = "SymLinks not enabled on Arm64 Windows runners"
)]
fn recursive_rmdir_of_symlink() {
    let tmpdir = tmpdir();
    let link = tmpdir.join("d1");
    let dir = tmpdir.join("d2");
    let canary = dir.join("do_not_delete");
    check!(fs::create_dir_all(&dir));
    check!(File::create(&canary));
    check!(junction_point(&dir, &link));
    check!(fs::remove_dir_all(&link));
    assert!(!link.is_dir());
    assert!(canary.exists());
}

#[test]
fn recursive_rmdir_of_file_fails() {
    let tmpdir = tmpdir();
    let canary = tmpdir.join("do_not_delete");
    check!(File::create(&canary));
    let result = fs::remove_dir_all(&canary);
    #[cfg(unix)]
    error!(result, "Not a directory");
    #[cfg(windows)]
    error!(result, 267);
    assert!(result.is_err());
    assert!(canary.exists());
}

#[test]
#[cfg(windows)]
fn recursive_rmdir_of_file_symlink() {
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    }
    let f1 = tmpdir.join("f1");
    let f2 = tmpdir.join("f2");
    check!(File::create(&f1));
    check!(symlink_file(&f1, &f2));
    assert!(fs::remove_dir_all(&f2).is_err());
}

#[test]
#[ignore]
fn recursive_rmdir_toctou() {
    let tmpdir = tmpdir();
    let victim_del_path = tmpdir.join("victim_del");
    let victim_del_path_clone = victim_del_path.clone();
    let attack_dest_dir = tmpdir.join("attack_dest");
    let attack_dest_file = attack_dest_dir.join("attack_file");
    fs::create_dir(&attack_dest_dir).unwrap();
    File::create(&attack_dest_file).unwrap();
    let drop_canary = Arc::new(());
    let weak = Arc::downgrade(&drop_canary);
    thread::spawn(move || {
        while weak.upgrade().is_some() {
            let _ = fs::remove_dir_all(&victim_del_path_clone);
        }
    });
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(1000) {
        if !attack_dest_file.exists() {
            panic!("Attack succeeded in {:?}", start.elapsed());
        }
        let _ = fs::create_dir(&victim_del_path);
        let _ = fs::remove_dir(&victim_del_path);
        let _ = symlink_dir(&attack_dest_dir, &victim_del_path);
    }
}

#[test]
fn unicode_path_is_dir() {
    assert!(Path::new(".").is_dir());
    let tmpdir = tmpdir();
    let mut dirpath = tmpdir.path().to_path_buf();
    dirpath.push("test-가一ー你好");
    check!(fs::create_dir(&dirpath));
    assert!(dirpath.is_dir());
}

#[test]
fn unicode_path_exists() {
    let tmpdir = tmpdir();
    let unicode = tmpdir.path().join("test-각丁ー再见");
    check!(fs::create_dir(&unicode));
    assert!(unicode.exists());
}

#[test]
fn copy_file_does_not_exist() {
    let from = Path::new("test/nonexistent-bogus-path");
    let to = Path::new("test/other-bogus-path");
    assert!(fs::copy(&from, &to).is_err());
    assert!(!from.exists());
    assert!(!to.exists());
}

#[test]
fn copy_src_does_not_exist() {
    let tmpdir = tmpdir();
    let from = Path::new("test/nonexistent-bogus-path");
    let to = tmpdir.join("out.txt");
    check!(File::create(&to).and_then(|mut f| f.write(b"hello")));
    assert!(fs::copy(&from, &to).is_err());
}

#[test]
fn copy_file_ok() {
    let tmpdir = tmpdir();
    let input = tmpdir.join("in.txt");
    let out = tmpdir.join("out.txt");
    check!(File::create(&input).and_then(|mut f| f.write(b"hello")));
    check!(fs::copy(&input, &out));
    assert_eq!(check!(fs::read(&out)), b"hello");
}

#[test]
fn copy_file_preserves_perm_bits() {
    let tmpdir = tmpdir();
    let input = tmpdir.join("in.txt");
    let out = tmpdir.join("out.txt");
    let mut perm = check!(File::create(&input)).metadata().unwrap().permissions();
    perm.set_readonly(true);
    check!(fs::set_permissions(&input, perm.clone()));
    check!(fs::copy(&input, &out));
    assert!(out.metadata().unwrap().permissions().readonly());
}

#[test]
fn symlinks_work() {
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    }
    let input = tmpdir.join("in.txt");
    let out = tmpdir.join("out.txt");
    check!(File::create(&input).and_then(|mut f| f.write(b"foobar")));
    check!(symlink_file(&input, &out));
    assert!(out.symlink_metadata().unwrap().file_type().is_symlink());
    assert_eq!(check!(fs::read(&out)), b"foobar");
}

#[test]
fn symlink_noexist() {
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    }
    check!(symlink_file("foo", tmpdir.join("bar")));
    assert_eq!(check!(fs::read_link(tmpdir.join("bar"))).to_str().unwrap(), "foo");
}

#[test]
fn read_link() {
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    }
    let link = tmpdir.join("link");
    check!(symlink_file("foo", &link));
    assert_eq!(check!(fs::read_link(&link)).to_str().unwrap(), "foo");
}

#[test]
#[cfg_attr(target_os = "android", ignore = "Android SELinux rules prevent creating hardlinks")]
fn links_work() {
    let tmpdir = tmpdir();
    let input = tmpdir.join("in.txt");
    let out = tmpdir.join("out.txt");
    check!(File::create(&input).and_then(|mut f| f.write(b"foobar")));
    check!(fs::hard_link(&input, &out));
    assert_eq!(check!(fs::read(&out)), b"foobar");
}

#[test]
fn chmod_works() {
    let tmpdir = tmpdir();
    let file = tmpdir.join("in.txt");
    check!(File::create(&file));
    let mut p = check!(file.metadata()).permissions();
    p.set_readonly(true);
    check!(file.set_permissions(p.clone()));
    assert!(file.metadata().unwrap().permissions().readonly());
}

#[test]
fn sync_doesnt_kill_anything() {
    let tmpdir = tmpdir();
    let path = tmpdir.join("in.txt");
    let mut file = check!(File::create(&path));
    check!(file.sync_all());
    check!(file.sync_data());
    check!(file.write(b"foo"));
    check!(file.sync_all());
}

#[test]
fn truncate_works() {
    let tmpdir = tmpdir();
    let path = tmpdir.join("in.txt");
    let mut file = check!(File::create(&path));
    check!(file.write(b"foo"));
    check!(file.set_len(10));
    check!(file.write(b"bar"));
    check!(file.set_len(2));
    check!(file.write(b"wut"));
    assert_eq!(check!(fs::read(&path)), b"fo\0\0\0\0wut");
}

#[test]
fn open_flavors() {
    let tmpdir = tmpdir();
    let invalid = "creating or truncating a file requires write or append access";
    let w = OpenOptions::new().write(true);
    let r = OpenOptions::new().read(true);
    let rw = OpenOptions::new().read(true).write(true);
    let a = OpenOptions::new().append(true);

    check!(w.clone().create_new(true).open(tmpdir.join("a")));
    check!(w.clone().create(true).truncate(true).open(tmpdir.join("a")));
    check!(w.clone().truncate(true).open(tmpdir.join("a")));
    check!(w.clone().create(true).open(tmpdir.join("a")));
    check!(w.open(tmpdir.join("a")));

    error_contains!(r.clone().create_new(true).open(tmpdir.join("b")), invalid);
    error_contains!(r.clone().create(true).truncate(true).open(tmpdir.join("b")), invalid);
    error_contains!(r.clone().truncate(true).open(tmpdir.join("b")), invalid);
    error_contains!(r.clone().create(true).open(tmpdir.join("b")), invalid);
    check!(r.open(tmpdir.join("a")));

    check!(rw.clone().create_new(true).open(tmpdir.join("c")));
    check!(rw.clone().create(true).truncate(true).open(tmpdir.join("c")));
    check!(rw.clone().truncate(true).open(tmpdir.join("c")));
    check!(rw.clone().create(true).open(tmpdir.join("c")));
    check!(rw.open(tmpdir.join("c")));

    check!(a.clone().create_new(true).open(tmpdir.join("d")));
    error_contains!(a.clone().create(true).truncate(true).open(tmpdir.join("d")), invalid);
    error_contains!(a.clone().truncate(true).open(tmpdir.join("d")), invalid);
    check!(a.clone().create(true).open(tmpdir.join("d")));
    check!(a.open(tmpdir.join("d")));
}

#[test]
fn binary_file() {
    let mut bytes = [0u8; 1024];
    crate::test_helpers::test_rng().fill_bytes(&mut bytes);
    let tmpdir = tmpdir();
    check!(fs::write(tmpdir.join("test"), &bytes));
    assert_eq!(check!(fs::read(tmpdir.join("test"))), bytes);
}

#[test]
fn write_then_read() {
    let mut bytes = [0u8; 1024];
    crate::test_helpers::test_rng().fill_bytes(&mut bytes);
    let tmpdir = tmpdir();
    check!(fs::write(tmpdir.join("test"), &bytes));
    assert_eq!(check!(fs::read(tmpdir.join("test"))), bytes);
    check!(fs::write(tmpdir.join("utf8"), "𐁁𐀓𐀠𐀴𐀍"));
    assert_eq!(check!(fs::read_to_string(tmpdir.join("utf8"))), "𐁁𐀓𐀠𐀴𐀍");
}

#[test]
fn file_try_clone() {
    let tmpdir = tmpdir();
    let mut f1 = check!(OpenOptions::new().read(true).write(true).create(true).open(tmpdir.join("test")));
    let mut f2 = check!(f1.try_clone());
    check!(f1.write_all(b"hello world"));
    check!(f1.seek(SeekFrom::Start(2)));
    let mut buf = Vec::new();
    check!(f2.read_to_end(&mut buf));
    assert_eq!(buf, b"llo world");
    drop(f2);
    check!(f1.write_all(b"!"));
}

#[test]
#[cfg(not(target_vendor = "win7"))]
fn unlink_readonly() {
    let tmpdir = tmpdir();
    let path = tmpdir.join("file");
    check!(File::create(&path));
    let mut perm = check!(path.metadata()).permissions();
    perm.set_readonly(true);
    check!(fs::set_permissions(&path, perm));
    check!(fs::remove_file(&path));
}

#[test]
fn mkdir_trailing_slash() {
    let tmpdir = tmpdir();
    let path = tmpdir.join("file/a/");
    check!(fs::create_dir_all(&path));
}

#[test]
fn canonicalize_works_simple() {
    let tmpdir = tmpdir();
    let file = tmpdir.join("test");
    check!(File::create(&file));
    assert_eq!(check!(fs::canonicalize(&file)), file);
}

#[test]
fn realpath_works() {
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    }
    let file = tmpdir.join("test");
    let dir = tmpdir.join("test2");
    let link = dir.join("link");
    let linkdir = tmpdir.join("test3");
    check!(File::create(&file));
    check!(fs::create_dir(&dir));
    check!(symlink_file(&file, &link));
    check!(symlink_dir(&dir, &linkdir));
    assert_eq!(check!(fs::canonicalize(&link)), file);
    assert_eq!(check!(fs::canonicalize(&linkdir)), dir);
}

#[test]
fn dir_entry_methods() {
    let tmpdir = tmpdir();
    check!(fs::create_dir(tmpdir.join("a")));
    check!(File::create(tmpdir.join("b")));
    for entry in check!(fs::read_dir(tmpdir.path())) {
        let entry = check!(entry);
        match entry.file_name().to_str() {
            Some("a") => assert!(entry.file_type().unwrap().is_dir()),
            Some("b") => assert!(entry.file_type().unwrap().is_file()),
            _ => panic!(),
        }
    }
}

#[test]
fn read_dir_not_found() {
    assert_eq!(fs::read_dir("/path/that/does/not/exist").unwrap_err().kind(), ErrorKind::NotFound);
}

#[test]
fn file_open_not_found() {
    assert_eq!(File::open("/path/that/does/not/exist").unwrap_err().kind(), ErrorKind::NotFound);
}

#[test]
#[cfg_attr(
    all(windows, target_arch = "aarch64"),
    ignore = "SymLinks not enabled on Arm64 Windows runners"
)]
fn create_dir_all_with_junctions() {
    let tmpdir = tmpdir();
    let target = tmpdir.join("target");
    let junction = tmpdir.join("junction");
    let b = junction.join("a/b");
    let link = tmpdir.join("link");
    let d = link.join("c/d");
    check!(fs::create_dir(&target));
    check!(junction_point(&target, &junction));
    check!(fs::create_dir_all(&b));
    assert!(b.exists());
    if got_symlink_permission(&tmpdir) {
        check!(symlink_dir(&target, &link));
        check!(fs::create_dir_all(&d));
        assert!(d.exists());
    }
}

#[test]
fn metadata_access_times() {
    let tmpdir = tmpdir();
    let file = tmpdir.join("file");
    check!(File::create(&file));
    let dir_meta = check!(fs::metadata(tmpdir.path()));
    let file_meta = check!(fs::metadata(&file));
    let _ = dir_meta.accessed();
    let _ = dir_meta.modified();
    let _ = file_meta.accessed();
    let _ = file_meta.modified();
}

#[test]
#[cfg_attr(target_os = "android", ignore = "Android SELinux rules prevent creating hardlinks")]
fn symlink_hard_link() {
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    }
    check!(File::create(tmpdir.join("file")));
    check!(symlink_file("file", tmpdir.join("symlink")));
    check!(fs::hard_link(tmpdir.join("symlink"), tmpdir.join("hard_link")));
    assert!(tmpdir.join("hard_link").symlink_metadata().unwrap().file_type().is_symlink());
}

#[test]
#[cfg(windows)]
fn create_dir_long_paths() {
    let tmpdir = tmpdir();
    let mut path = tmpdir.path().to_path_buf();
    path.push("a");
    let mut path = path.into_os_string();
    path.push(std::iter::repeat("a").take(240).collect::<String>());
    fs::create_dir(&path).unwrap();
}

#[test]
fn read_large_dir() {
    let tmpdir = tmpdir();
    for i in 0..32_768 {
        check!(File::create(tmpdir.join(i.to_string())));
    }
    for entry in fs::read_dir(tmpdir.path()).unwrap() {
        entry.unwrap();
    }
}

#[test]
fn test_eq_direntry_metadata() {
    let tmpdir = tmpdir();
    check!(File::create(tmpdir.join("file")));
    for e in fs::read_dir(tmpdir.path()).unwrap() {
        let e = e.unwrap();
        assert_eq!(e.file_type().unwrap(), e.path().metadata().unwrap().file_type());
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_read_dir_infinite_loop() {
    use std::process::Command;
    let mut child = match Command::new("echo").spawn() {
        Ok(c) => c,
        Err(_) => return,
    };
    let _ = child.kill();
    let path = format!("/proc/{}/net", child.id());
    if fs::read_dir(path).is_err() {
        return;
    }
    let dir = fs::read_dir(path).unwrap();
    assert!(dir.filter(|e| e.is_err()).take(2).count() < 2);
}

#[test]
fn rename_directory() {
    let tmpdir = tmpdir();
    let old = tmpdir.join("foo/bar/baz");
    let new = tmpdir.join("quux/blat/newdir");
    check!(fs::create_dir_all(&old));
    check!(File::create(old.join("temp.txt")));
    check!(fs::create_dir_all(tmpdir.join("quux/blat")));
    check!(fs::rename(&old, &new));
    assert!(new.is_dir());
    assert!(new.join("temp.txt").exists());
}

#[test]
fn test_file_times() {
    let tmp = tmpdir();
    let file = check!(File::create(tmp.join("foo")));
    let times = FileTimes::new()
        .set_accessed(SystemTime::UNIX_EPOCH + Duration::from_secs(12345))
        .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(54321));
    let _ = file.set_times(times);
    let meta = file.metadata().unwrap();
    assert_eq!(meta.accessed().unwrap(), SystemTime::UNIX_EPOCH + Duration::from_secs(12345));
    assert_eq!(meta.modified().unwrap(), SystemTime::UNIX_EPOCH + Duration::from_secs(54321));
}

#[test]
#[cfg(windows)]
fn test_hidden_file_truncation() {
    let tmpdir = tmpdir();
    let path = tmpdir.join("hidden_file.txt");
    const FILE_ATTRIBUTE_HIDDEN: u32 = 2;
    let mut file = check!(OpenOptions::new()
        .write(true)
        .create_new(true)
        .attributes(FILE_ATTRIBUTE_HIDDEN)
        .open(&path));
    check!(file.write(b"hidden world!"));
    drop(file);
    let file = check!(File::create(&path));
    assert_eq!(file.metadata().unwrap().len(), 0);
}

#[test]
#[cfg(windows)]
#[cfg_attr(target_vendor = "win7", ignore = "Unsupported under Windows 7")]
fn test_rename_file_over_open_file() {
    let tmpdir = tmpdir();
    let source = tmpdir.join("source_file.txt");
    let target = tmpdir.join("target_file.txt");
    check!(fs::write(&source, b"source hello world"));
    check!(fs::write(&target, b"target hello world"));
    let _handle = check!(File::open(&target));
    check!(fs::rename(&source, &target));
    assert_eq!(check!(fs::read(&target)), b"source hello world");
}

#[test]
#[cfg(windows)]
#[cfg_attr(target_vendor = "win7", ignore = "Unsupported under Windows 7")]
fn test_rename_directory_to_non_empty_directory() {
    let tmpdir = tmpdir();
    let source = tmpdir.join("source_directory");
    let target = tmpdir.join("target_directory");
    check!(fs::create_dir(&source));
    check!(fs::create_dir(&target));
    check!(fs::write(target.join("target_file.txt"), b"target hello world"));
    error!(fs::rename(&source, &target), 145);
}

#[test]
fn test_rename_symlink() {
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    }
    let original = tmpdir.join("original");
    let dest = tmpdir.join("dest");
    check!(symlink_file("does not exist", &original));
    check!(fs::rename(&original, &dest));
    assert_eq!(check!(fs::read_link(&dest)).to_str().unwrap(), "does not exist");
}

#[test]
#[cfg(windows)]
#[cfg_attr(
    all(windows, target_arch = "aarch64"),
    ignore = "SymLinks not enabled on Arm64 Windows runners"
)]
fn test_rename_junction() {
    let tmpdir = tmpdir();
    let original = tmpdir.join("original");
    let dest = tmpdir.join("dest");
    check!(junction_point("does not exist", &original));
    check!(fs::rename(&original, &dest));
    assert!(check!(fs::read_link(&dest)).file_name().is_some());
}

#[test]
fn test_open_options_invalid_combinations() {
    let msg = "creating or truncating a file requires write or append access";
    error_contains!(OpenOptions::new().read(true).create(true).open("x"), msg);
    error_contains!(OpenOptions::new().read(true).create_new(true).open("x"), msg);
    error_contains!(OpenOptions::new().read(true).truncate(true).open("x"), msg);
    error_contains!(OpenOptions::new().append(true).truncate(true).open("x"), msg);
    assert!(OpenOptions::new().open("x").is_err());
}

#[test]
fn test_fs_set_times() {
    let tmp = tmpdir();
    let path = tmp.join("foo");
    check!(File::create(&path));
    let times = FileTimes::new()
        .set_accessed(SystemTime::UNIX_EPOCH + Duration::from_secs(12345))
        .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(54321));
    let _ = fs::set_times(&path, times);
}

#[test]
fn test_fs_set_times_follows_symlink() {
    let tmp = tmpdir();
    let target = tmp.join("target");
    let link = tmp.join("link");
    check!(File::create(&target));
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &link).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(&target, &link).unwrap();
    let times = FileTimes::new()
        .set_accessed(SystemTime::UNIX_EPOCH + Duration::from_secs(12345))
        .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(54321));
    let _ = fs::set_times(&link, times);
    assert_eq!(fs::metadata(&target).unwrap().modified().unwrap(), SystemTime::UNIX_EPOCH + Duration::from_secs(54321));
}

#[test]
fn test_fs_set_times_nofollow() {
    let tmp = tmpdir();
    let target = tmp.join("target");
    let link = tmp.join("link");
    check!(File::create(&target));
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &link).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(&target, &link).unwrap();
    let times = FileTimes::new()
        .set_accessed(SystemTime::UNIX_EPOCH + Duration::from_secs(11111))
        .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(22222));
    let _ = fs::set_times_nofollow(&link, times);
    assert_eq!(fs::symlink_metadata(&link).unwrap().modified().unwrap(), SystemTime::UNIX_EPOCH + Duration::from_secs(22222));
}