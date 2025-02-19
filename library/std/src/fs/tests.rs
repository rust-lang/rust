use rand::RngCore;

use crate::char::MAX_LEN_UTF8;
use crate::fs::{self, File, FileTimes, OpenOptions};
use crate::io::prelude::*;
use crate::io::{BorrowedBuf, ErrorKind, SeekFrom};
use crate::mem::MaybeUninit;
#[cfg(unix)]
use crate::os::unix::fs::symlink as symlink_dir;
#[cfg(unix)]
use crate::os::unix::fs::symlink as symlink_file;
#[cfg(unix)]
use crate::os::unix::fs::symlink as junction_point;
#[cfg(windows)]
use crate::os::windows::fs::{OpenOptionsExt, junction_point, symlink_dir, symlink_file};
use crate::path::Path;
use crate::sync::Arc;
use crate::test_helpers::{TempDir, tmpdir};
use crate::time::{Duration, Instant, SystemTime};
use crate::{env, str, thread};

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
                assert!(err.raw_os_error() == Some($s), "`{}` did not have a code of `{}`", err, $s)
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
                assert!(err.to_string().contains($s), "`{}` did not contain `{}`", err, $s)
            }
        }
    };
}

// Several test fail on windows if the user does not have permission to
// create symlinks (the `SeCreateSymbolicLinkPrivilege`). Instead of
// disabling these test on Windows, use this function to test whether we
// have permission, and return otherwise. This way, we still don't run these
// tests most of the time, but at least we do if the user has the right
// permissions.
pub fn got_symlink_permission(tmpdir: &TempDir) -> bool {
    if cfg!(not(windows)) || env::var_os("CI").is_some() {
        return true;
    }
    let link = tmpdir.join("some_hopefully_unique_link_name");

    match symlink_file(r"nonexisting_target", link) {
        // ERROR_PRIVILEGE_NOT_HELD = 1314
        Err(ref err) if err.raw_os_error() == Some(1314) => false,
        Ok(_) | Err(_) => true,
    }
}

#[test]
fn file_test_io_smoke_test() {
    let message = "it's alright. have a good time";
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_rt_io_file_test.txt");
    {
        let mut write_stream = check!(File::create(filename));
        check!(write_stream.write(message.as_bytes()));
    }
    {
        let mut read_stream = check!(File::open(filename));
        let mut read_buf = [0; 1028];
        let read_str = match check!(read_stream.read(&mut read_buf)) {
            0 => panic!("shouldn't happen"),
            n => str::from_utf8(&read_buf[..n]).unwrap().to_string(),
        };
        assert_eq!(read_str, message);
    }
    check!(fs::remove_file(filename));
}

#[test]
fn invalid_path_raises() {
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_that_does_not_exist.txt");
    let result = File::open(filename);

    #[cfg(all(unix, not(target_os = "vxworks")))]
    error!(result, "No such file or directory");
    #[cfg(target_os = "vxworks")]
    error!(result, "no such file or directory");
    #[cfg(windows)]
    error!(result, 2); // ERROR_FILE_NOT_FOUND
}

#[test]
fn file_test_iounlinking_invalid_path_should_raise_condition() {
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_another_file_that_does_not_exist.txt");

    let result = fs::remove_file(filename);

    #[cfg(all(unix, not(target_os = "vxworks")))]
    error!(result, "No such file or directory");
    #[cfg(target_os = "vxworks")]
    error!(result, "no such file or directory");
    #[cfg(windows)]
    error!(result, 2); // ERROR_FILE_NOT_FOUND
}

#[test]
fn file_test_io_non_positional_read() {
    let message: &str = "ten-four";
    let mut read_mem = [0; 8];
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_rt_io_file_test_positional.txt");
    {
        let mut rw_stream = check!(File::create(filename));
        check!(rw_stream.write(message.as_bytes()));
    }
    {
        let mut read_stream = check!(File::open(filename));
        {
            let read_buf = &mut read_mem[0..4];
            check!(read_stream.read(read_buf));
        }
        {
            let read_buf = &mut read_mem[4..8];
            check!(read_stream.read(read_buf));
        }
    }
    check!(fs::remove_file(filename));
    let read_str = str::from_utf8(&read_mem).unwrap();
    assert_eq!(read_str, message);
}

#[test]
fn file_test_io_seek_and_tell_smoke_test() {
    let message = "ten-four";
    let mut read_mem = [0; MAX_LEN_UTF8];
    let set_cursor = 4 as u64;
    let tell_pos_pre_read;
    let tell_pos_post_read;
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_rt_io_file_test_seeking.txt");
    {
        let mut rw_stream = check!(File::create(filename));
        check!(rw_stream.write(message.as_bytes()));
    }
    {
        let mut read_stream = check!(File::open(filename));
        check!(read_stream.seek(SeekFrom::Start(set_cursor)));
        tell_pos_pre_read = check!(read_stream.stream_position());
        check!(read_stream.read(&mut read_mem));
        tell_pos_post_read = check!(read_stream.stream_position());
    }
    check!(fs::remove_file(filename));
    let read_str = str::from_utf8(&read_mem).unwrap();
    assert_eq!(read_str, &message[4..8]);
    assert_eq!(tell_pos_pre_read, set_cursor);
    assert_eq!(tell_pos_post_read, message.len() as u64);
}

#[test]
fn file_test_io_seek_and_write() {
    let initial_msg = "food-is-yummy";
    let overwrite_msg = "-the-bar!!";
    let final_msg = "foo-the-bar!!";
    let seek_idx = 3;
    let mut read_mem = [0; 13];
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_rt_io_file_test_seek_and_write.txt");
    {
        let mut rw_stream = check!(File::create(filename));
        check!(rw_stream.write(initial_msg.as_bytes()));
        check!(rw_stream.seek(SeekFrom::Start(seek_idx)));
        check!(rw_stream.write(overwrite_msg.as_bytes()));
    }
    {
        let mut read_stream = check!(File::open(filename));
        check!(read_stream.read(&mut read_mem));
    }
    check!(fs::remove_file(filename));
    let read_str = str::from_utf8(&read_mem).unwrap();
    assert!(read_str == final_msg);
}

#[test]
#[cfg(any(
    windows,
    target_os = "freebsd",
    target_os = "linux",
    target_os = "netbsd",
    target_vendor = "apple",
))]
fn file_lock_multiple_shared() {
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_lock_multiple_shared_test.txt");
    let f1 = check!(File::create(filename));
    let f2 = check!(OpenOptions::new().write(true).open(filename));

    // Check that we can acquire concurrent shared locks
    check!(f1.lock_shared());
    check!(f2.lock_shared());
    check!(f1.unlock());
    check!(f2.unlock());
    assert!(check!(f1.try_lock_shared()));
    assert!(check!(f2.try_lock_shared()));
}

#[test]
#[cfg(any(
    windows,
    target_os = "freebsd",
    target_os = "linux",
    target_os = "netbsd",
    target_vendor = "apple",
))]
fn file_lock_blocking() {
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_lock_blocking_test.txt");
    let f1 = check!(File::create(filename));
    let f2 = check!(OpenOptions::new().write(true).open(filename));

    // Check that shared locks block exclusive locks
    check!(f1.lock_shared());
    assert!(!check!(f2.try_lock()));
    check!(f1.unlock());

    // Check that exclusive locks block shared locks
    check!(f1.lock());
    assert!(!check!(f2.try_lock_shared()));
}

#[test]
#[cfg(any(
    windows,
    target_os = "freebsd",
    target_os = "linux",
    target_os = "netbsd",
    target_vendor = "apple",
))]
fn file_lock_drop() {
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_lock_dup_test.txt");
    let f1 = check!(File::create(filename));
    let f2 = check!(OpenOptions::new().write(true).open(filename));

    // Check that locks are released when the File is dropped
    check!(f1.lock_shared());
    assert!(!check!(f2.try_lock()));
    drop(f1);
    assert!(check!(f2.try_lock()));
}

#[test]
#[cfg(any(
    windows,
    target_os = "freebsd",
    target_os = "linux",
    target_os = "netbsd",
    target_vendor = "apple",
))]
fn file_lock_dup() {
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_lock_dup_test.txt");
    let f1 = check!(File::create(filename));
    let f2 = check!(OpenOptions::new().write(true).open(filename));

    // Check that locks are not dropped if the File has been cloned
    check!(f1.lock_shared());
    assert!(!check!(f2.try_lock()));
    let cloned = check!(f1.try_clone());
    drop(f1);
    assert!(!check!(f2.try_lock()));
    drop(cloned)
}

#[test]
#[cfg(windows)]
fn file_lock_double_unlock() {
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_lock_double_unlock_test.txt");
    let f1 = check!(File::create(filename));
    let f2 = check!(OpenOptions::new().write(true).open(filename));

    // On Windows a file handle may acquire both a shared and exclusive lock.
    // Check that both are released by unlock()
    check!(f1.lock());
    check!(f1.lock_shared());
    assert!(!check!(f2.try_lock()));
    check!(f1.unlock());
    assert!(check!(f2.try_lock()));
}

#[test]
#[cfg(windows)]
fn file_lock_blocking_async() {
    use crate::thread::{sleep, spawn};
    const FILE_FLAG_OVERLAPPED: u32 = 0x40000000;

    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_lock_blocking_async.txt");
    let f1 = check!(File::create(filename));
    let f2 =
        check!(OpenOptions::new().custom_flags(FILE_FLAG_OVERLAPPED).write(true).open(filename));

    check!(f1.lock());

    // Ensure that lock() is synchronous when the file is opened for asynchronous IO
    let t = spawn(move || {
        check!(f2.lock());
    });
    sleep(Duration::from_secs(1));
    assert!(!t.is_finished());
    check!(f1.unlock());
    t.join().unwrap();

    // Ensure that lock_shared() is synchronous when the file is opened for asynchronous IO
    let f2 =
        check!(OpenOptions::new().custom_flags(FILE_FLAG_OVERLAPPED).write(true).open(filename));
    check!(f1.lock());

    // Ensure that lock() is synchronous when the file is opened for asynchronous IO
    let t = spawn(move || {
        check!(f2.lock_shared());
    });
    sleep(Duration::from_secs(1));
    assert!(!t.is_finished());
    check!(f1.unlock());
    t.join().unwrap();
}

#[test]
fn file_test_io_seek_shakedown() {
    //                   01234567890123
    let initial_msg = "qwer-asdf-zxcv";
    let chunk_one: &str = "qwer";
    let chunk_two: &str = "asdf";
    let chunk_three: &str = "zxcv";
    let mut read_mem = [0; MAX_LEN_UTF8];
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_rt_io_file_test_seek_shakedown.txt");
    {
        let mut rw_stream = check!(File::create(filename));
        check!(rw_stream.write(initial_msg.as_bytes()));
    }
    {
        let mut read_stream = check!(File::open(filename));

        check!(read_stream.seek(SeekFrom::End(-4)));
        check!(read_stream.read(&mut read_mem));
        assert_eq!(str::from_utf8(&read_mem).unwrap(), chunk_three);

        check!(read_stream.seek(SeekFrom::Current(-9)));
        check!(read_stream.read(&mut read_mem));
        assert_eq!(str::from_utf8(&read_mem).unwrap(), chunk_two);

        check!(read_stream.seek(SeekFrom::Start(0)));
        check!(read_stream.read(&mut read_mem));
        assert_eq!(str::from_utf8(&read_mem).unwrap(), chunk_one);
    }
    check!(fs::remove_file(filename));
}

#[test]
fn file_test_io_eof() {
    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_eof.txt");
    let mut buf = [0; 256];
    {
        let oo = OpenOptions::new().create_new(true).write(true).read(true).clone();
        let mut rw = check!(oo.open(&filename));
        assert_eq!(check!(rw.read(&mut buf)), 0);
        assert_eq!(check!(rw.read(&mut buf)), 0);
    }
    check!(fs::remove_file(&filename));
}

#[test]
#[cfg(unix)]
fn file_test_io_read_write_at() {
    use crate::os::unix::fs::FileExt;

    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_read_write_at.txt");
    let mut buf = [0; 256];
    let write1 = "asdf";
    let write2 = "qwer-";
    let write3 = "-zxcv";
    let content = "qwer-asdf-zxcv";
    {
        let oo = OpenOptions::new().create_new(true).write(true).read(true).clone();
        let mut rw = check!(oo.open(&filename));
        assert_eq!(check!(rw.write_at(write1.as_bytes(), 5)), write1.len());
        assert_eq!(check!(rw.stream_position()), 0);
        assert_eq!(check!(rw.read_at(&mut buf, 5)), write1.len());
        assert_eq!(str::from_utf8(&buf[..write1.len()]), Ok(write1));
        assert_eq!(check!(rw.stream_position()), 0);
        assert_eq!(check!(rw.read_at(&mut buf[..write2.len()], 0)), write2.len());
        assert_eq!(str::from_utf8(&buf[..write2.len()]), Ok("\0\0\0\0\0"));
        assert_eq!(check!(rw.stream_position()), 0);
        assert_eq!(check!(rw.write(write2.as_bytes())), write2.len());
        assert_eq!(check!(rw.stream_position()), 5);
        assert_eq!(check!(rw.read(&mut buf)), write1.len());
        assert_eq!(str::from_utf8(&buf[..write1.len()]), Ok(write1));
        assert_eq!(check!(rw.stream_position()), 9);
        assert_eq!(check!(rw.read_at(&mut buf[..write2.len()], 0)), write2.len());
        assert_eq!(str::from_utf8(&buf[..write2.len()]), Ok(write2));
        assert_eq!(check!(rw.stream_position()), 9);
        assert_eq!(check!(rw.write_at(write3.as_bytes(), 9)), write3.len());
        assert_eq!(check!(rw.stream_position()), 9);
    }
    {
        let mut read = check!(File::open(&filename));
        assert_eq!(check!(read.read_at(&mut buf, 0)), content.len());
        assert_eq!(str::from_utf8(&buf[..content.len()]), Ok(content));
        assert_eq!(check!(read.stream_position()), 0);
        assert_eq!(check!(read.seek(SeekFrom::End(-5))), 9);
        assert_eq!(check!(read.read_at(&mut buf, 0)), content.len());
        assert_eq!(str::from_utf8(&buf[..content.len()]), Ok(content));
        assert_eq!(check!(read.stream_position()), 9);
        assert_eq!(check!(read.read(&mut buf)), write3.len());
        assert_eq!(str::from_utf8(&buf[..write3.len()]), Ok(write3));
        assert_eq!(check!(read.stream_position()), 14);
        assert_eq!(check!(read.read_at(&mut buf, 0)), content.len());
        assert_eq!(str::from_utf8(&buf[..content.len()]), Ok(content));
        assert_eq!(check!(read.stream_position()), 14);
        assert_eq!(check!(read.read_at(&mut buf, 14)), 0);
        assert_eq!(check!(read.read_at(&mut buf, 15)), 0);
        assert_eq!(check!(read.stream_position()), 14);
    }
    check!(fs::remove_file(&filename));
}

#[test]
#[cfg(unix)]
fn set_get_unix_permissions() {
    use crate::os::unix::fs::PermissionsExt;

    let tmpdir = tmpdir();
    let filename = &tmpdir.join("set_get_unix_permissions");
    check!(fs::create_dir(filename));
    let mask = 0o7777;

    check!(fs::set_permissions(filename, fs::Permissions::from_mode(0)));
    let metadata0 = check!(fs::metadata(filename));
    assert_eq!(mask & metadata0.permissions().mode(), 0);

    check!(fs::set_permissions(filename, fs::Permissions::from_mode(0o1777)));
    let metadata1 = check!(fs::metadata(filename));
    #[cfg(all(unix, not(target_os = "vxworks")))]
    assert_eq!(mask & metadata1.permissions().mode(), 0o1777);
    #[cfg(target_os = "vxworks")]
    assert_eq!(mask & metadata1.permissions().mode(), 0o0777);
}

#[test]
#[cfg(windows)]
fn file_test_io_seek_read_write() {
    use crate::os::windows::fs::FileExt;

    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_rt_io_file_test_seek_read_write.txt");
    let mut buf = [0; 256];
    let write1 = "asdf";
    let write2 = "qwer-";
    let write3 = "-zxcv";
    let content = "qwer-asdf-zxcv";
    {
        let oo = OpenOptions::new().create_new(true).write(true).read(true).clone();
        let mut rw = check!(oo.open(&filename));
        assert_eq!(check!(rw.seek_write(write1.as_bytes(), 5)), write1.len());
        assert_eq!(check!(rw.stream_position()), 9);
        assert_eq!(check!(rw.seek_read(&mut buf, 5)), write1.len());
        assert_eq!(str::from_utf8(&buf[..write1.len()]), Ok(write1));
        assert_eq!(check!(rw.stream_position()), 9);
        assert_eq!(check!(rw.seek(SeekFrom::Start(0))), 0);
        assert_eq!(check!(rw.write(write2.as_bytes())), write2.len());
        assert_eq!(check!(rw.stream_position()), 5);
        assert_eq!(check!(rw.read(&mut buf)), write1.len());
        assert_eq!(str::from_utf8(&buf[..write1.len()]), Ok(write1));
        assert_eq!(check!(rw.stream_position()), 9);
        assert_eq!(check!(rw.seek_read(&mut buf[..write2.len()], 0)), write2.len());
        assert_eq!(str::from_utf8(&buf[..write2.len()]), Ok(write2));
        assert_eq!(check!(rw.stream_position()), 5);
        assert_eq!(check!(rw.seek_write(write3.as_bytes(), 9)), write3.len());
        assert_eq!(check!(rw.stream_position()), 14);
    }
    {
        let mut read = check!(File::open(&filename));
        assert_eq!(check!(read.seek_read(&mut buf, 0)), content.len());
        assert_eq!(str::from_utf8(&buf[..content.len()]), Ok(content));
        assert_eq!(check!(read.stream_position()), 14);
        assert_eq!(check!(read.seek(SeekFrom::End(-5))), 9);
        assert_eq!(check!(read.seek_read(&mut buf, 0)), content.len());
        assert_eq!(str::from_utf8(&buf[..content.len()]), Ok(content));
        assert_eq!(check!(read.stream_position()), 14);
        assert_eq!(check!(read.seek(SeekFrom::End(-5))), 9);
        assert_eq!(check!(read.read(&mut buf)), write3.len());
        assert_eq!(str::from_utf8(&buf[..write3.len()]), Ok(write3));
        assert_eq!(check!(read.stream_position()), 14);
        assert_eq!(check!(read.seek_read(&mut buf, 0)), content.len());
        assert_eq!(str::from_utf8(&buf[..content.len()]), Ok(content));
        assert_eq!(check!(read.stream_position()), 14);
        assert_eq!(check!(read.seek_read(&mut buf, 14)), 0);
        assert_eq!(check!(read.seek_read(&mut buf, 15)), 0);
    }
    check!(fs::remove_file(&filename));
}

#[test]
fn file_test_read_buf() {
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("test");
    check!(fs::write(filename, &[1, 2, 3, 4]));

    let mut buf: [MaybeUninit<u8>; 128] = [MaybeUninit::uninit(); 128];
    let mut buf = BorrowedBuf::from(buf.as_mut_slice());
    let mut file = check!(File::open(filename));
    check!(file.read_buf(buf.unfilled()));
    assert_eq!(buf.filled(), &[1, 2, 3, 4]);
    // File::read_buf should omit buffer initialization.
    assert_eq!(buf.init_len(), 4);

    check!(fs::remove_file(filename));
}

#[test]
fn file_test_stat_is_correct_on_is_file() {
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_stat_correct_on_is_file.txt");
    {
        let mut opts = OpenOptions::new();
        let mut fs = check!(opts.read(true).write(true).create(true).open(filename));
        let msg = "hw";
        fs.write(msg.as_bytes()).unwrap();

        let fstat_res = check!(fs.metadata());
        assert!(fstat_res.is_file());
    }
    let stat_res_fn = check!(fs::metadata(filename));
    assert!(stat_res_fn.is_file());
    let stat_res_meth = check!(filename.metadata());
    assert!(stat_res_meth.is_file());
    check!(fs::remove_file(filename));
}

#[test]
fn file_test_stat_is_correct_on_is_dir() {
    let tmpdir = tmpdir();
    let filename = &tmpdir.join("file_stat_correct_on_is_dir");
    check!(fs::create_dir(filename));
    let stat_res_fn = check!(fs::metadata(filename));
    assert!(stat_res_fn.is_dir());
    let stat_res_meth = check!(filename.metadata());
    assert!(stat_res_meth.is_dir());
    check!(fs::remove_dir(filename));
}

#[test]
fn file_test_fileinfo_false_when_checking_is_file_on_a_directory() {
    let tmpdir = tmpdir();
    let dir = &tmpdir.join("fileinfo_false_on_dir");
    check!(fs::create_dir(dir));
    assert!(!dir.is_file());
    check!(fs::remove_dir(dir));
}

#[test]
fn file_test_fileinfo_check_exists_before_and_after_file_creation() {
    let tmpdir = tmpdir();
    let file = &tmpdir.join("fileinfo_check_exists_b_and_a.txt");
    check!(check!(File::create(file)).write(b"foo"));
    assert!(file.exists());
    check!(fs::remove_file(file));
    assert!(!file.exists());
}

#[test]
fn file_test_directoryinfo_check_exists_before_and_after_mkdir() {
    let tmpdir = tmpdir();
    let dir = &tmpdir.join("before_and_after_dir");
    assert!(!dir.exists());
    check!(fs::create_dir(dir));
    assert!(dir.exists());
    assert!(dir.is_dir());
    check!(fs::remove_dir(dir));
    assert!(!dir.exists());
}

#[test]
fn file_test_directoryinfo_readdir() {
    let tmpdir = tmpdir();
    let dir = &tmpdir.join("di_readdir");
    check!(fs::create_dir(dir));
    let prefix = "foo";
    for n in 0..3 {
        let f = dir.join(&format!("{n}.txt"));
        let mut w = check!(File::create(&f));
        let msg_str = format!("{}{}", prefix, n.to_string());
        let msg = msg_str.as_bytes();
        check!(w.write(msg));
    }
    let files = check!(fs::read_dir(dir));
    let mut mem = [0; MAX_LEN_UTF8];
    for f in files {
        let f = f.unwrap().path();
        {
            let n = f.file_stem().unwrap();
            check!(check!(File::open(&f)).read(&mut mem));
            let read_str = str::from_utf8(&mem).unwrap();
            let expected = format!("{}{}", prefix, n.to_str().unwrap());
            assert_eq!(expected, read_str);
        }
        check!(fs::remove_file(&f));
    }
    check!(fs::remove_dir(dir));
}

#[test]
fn file_create_new_already_exists_error() {
    let tmpdir = tmpdir();
    let file = &tmpdir.join("file_create_new_error_exists");
    check!(fs::File::create(file));
    let e = fs::OpenOptions::new().write(true).create_new(true).open(file).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::AlreadyExists);
}

#[test]
fn mkdir_path_already_exists_error() {
    let tmpdir = tmpdir();
    let dir = &tmpdir.join("mkdir_error_twice");
    check!(fs::create_dir(dir));
    let e = fs::create_dir(dir).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::AlreadyExists);
}

#[test]
fn recursive_mkdir() {
    let tmpdir = tmpdir();
    let dir = tmpdir.join("d1/d2");
    check!(fs::create_dir_all(&dir));
    assert!(dir.is_dir())
}

#[test]
fn recursive_mkdir_failure() {
    let tmpdir = tmpdir();
    let dir = tmpdir.join("d1");
    let file = dir.join("f1");

    check!(fs::create_dir_all(&dir));
    check!(File::create(&file));

    let result = fs::create_dir_all(&file);

    assert!(result.is_err());
}

#[test]
fn concurrent_recursive_mkdir() {
    for _ in 0..100 {
        let dir = tmpdir();
        let mut dir = dir.join("a");
        for _ in 0..40 {
            dir = dir.join("a");
        }
        let mut join = vec![];
        for _ in 0..8 {
            let dir = dir.clone();
            join.push(thread::spawn(move || {
                check!(fs::create_dir_all(&dir));
            }))
        }

        // No `Display` on result of `join()`
        join.drain(..).map(|join| join.join().unwrap()).count();
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
fn recursive_rmdir() {
    let tmpdir = tmpdir();
    let d1 = tmpdir.join("d1");
    let dt = d1.join("t");
    let dtt = dt.join("t");
    let d2 = tmpdir.join("d2");
    let canary = d2.join("do_not_delete");
    check!(fs::create_dir_all(&dtt));
    check!(fs::create_dir_all(&d2));
    check!(check!(File::create(&canary)).write(b"foo"));
    check!(junction_point(&d2, &dt.join("d2")));
    let _ = symlink_file(&canary, &d1.join("canary"));
    check!(fs::remove_dir_all(&d1));

    assert!(!d1.is_dir());
    assert!(canary.exists());
}

#[test]
fn recursive_rmdir_of_symlink() {
    // test we do not recursively delete a symlink but only dirs.
    let tmpdir = tmpdir();
    let link = tmpdir.join("d1");
    let dir = tmpdir.join("d2");
    let canary = dir.join("do_not_delete");
    check!(fs::create_dir_all(&dir));
    check!(check!(File::create(&canary)).write(b"foo"));
    check!(junction_point(&dir, &link));
    check!(fs::remove_dir_all(&link));

    assert!(!link.is_dir());
    assert!(canary.exists());
}

#[test]
fn recursive_rmdir_of_file_fails() {
    // test we do not delete a directly specified file.
    let tmpdir = tmpdir();
    let canary = tmpdir.join("do_not_delete");
    check!(check!(File::create(&canary)).write(b"foo"));
    let result = fs::remove_dir_all(&canary);
    #[cfg(unix)]
    error!(result, "Not a directory");
    #[cfg(windows)]
    error!(result, 267); // ERROR_DIRECTORY - The directory name is invalid.
    assert!(result.is_err());
    assert!(canary.exists());
}

#[test]
// only Windows makes a distinction between file and directory symlinks.
#[cfg(windows)]
fn recursive_rmdir_of_file_symlink() {
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    };

    let f1 = tmpdir.join("f1");
    let f2 = tmpdir.join("f2");
    check!(check!(File::create(&f1)).write(b"foo"));
    check!(symlink_file(&f1, &f2));
    match fs::remove_dir_all(&f2) {
        Ok(..) => panic!("wanted a failure"),
        Err(..) => {}
    }
}

#[test]
#[ignore] // takes too much time
fn recursive_rmdir_toctou() {
    // Test for time-of-check to time-of-use issues.
    //
    // Scenario:
    // The attacker wants to get directory contents deleted, to which they do not have access.
    // They have a way to get a privileged Rust binary call `std::fs::remove_dir_all()` on a
    // directory they control, e.g. in their home directory.
    //
    // The POC sets up the `attack_dest/attack_file` which the attacker wants to have deleted.
    // The attacker repeatedly creates a directory and replaces it with a symlink from
    // `victim_del` to `attack_dest` while the victim code calls `std::fs::remove_dir_all()`
    // on `victim_del`. After a few seconds the attack has succeeded and
    // `attack_dest/attack_file` is deleted.
    let tmpdir = tmpdir();
    let victim_del_path = tmpdir.join("victim_del");
    let victim_del_path_clone = victim_del_path.clone();

    // setup dest
    let attack_dest_dir = tmpdir.join("attack_dest");
    let attack_dest_dir = attack_dest_dir.as_path();
    fs::create_dir(attack_dest_dir).unwrap();
    let attack_dest_file = tmpdir.join("attack_dest/attack_file");
    File::create(&attack_dest_file).unwrap();

    let drop_canary_arc = Arc::new(());
    let drop_canary_weak = Arc::downgrade(&drop_canary_arc);

    eprintln!("x: {victim_del_path:?}");

    // victim just continuously removes `victim_del`
    thread::spawn(move || {
        while drop_canary_weak.upgrade().is_some() {
            let _ = fs::remove_dir_all(&victim_del_path_clone);
        }
    });

    // attacker (could of course be in a separate process)
    let start_time = Instant::now();
    while Instant::now().duration_since(start_time) < Duration::from_secs(1000) {
        if !attack_dest_file.exists() {
            panic!(
                "Victim deleted symlinked file outside of victim_del. Attack succeeded in {:?}.",
                Instant::now().duration_since(start_time)
            );
        }
        let _ = fs::create_dir(&victim_del_path);
        let _ = fs::remove_dir(&victim_del_path);
        let _ = symlink_dir(attack_dest_dir, &victim_del_path);
    }
}

#[test]
fn unicode_path_is_dir() {
    assert!(Path::new(".").is_dir());
    assert!(!Path::new("test/stdtest/fs.rs").is_dir());

    let tmpdir = tmpdir();

    let mut dirpath = tmpdir.path().to_path_buf();
    dirpath.push("test-가一ー你好");
    check!(fs::create_dir(&dirpath));
    assert!(dirpath.is_dir());

    let mut filepath = dirpath;
    filepath.push("unicode-file-\u{ac00}\u{4e00}\u{30fc}\u{4f60}\u{597d}.rs");
    check!(File::create(&filepath)); // ignore return; touch only
    assert!(!filepath.is_dir());
    assert!(filepath.exists());
}

#[test]
fn unicode_path_exists() {
    assert!(Path::new(".").exists());
    assert!(!Path::new("test/nonexistent-bogus-path").exists());

    let tmpdir = tmpdir();
    let unicode = tmpdir.path();
    let unicode = unicode.join("test-각丁ー再见");
    check!(fs::create_dir(&unicode));
    assert!(unicode.exists());
    assert!(!Path::new("test/unicode-bogus-path-각丁ー再见").exists());
}

#[test]
fn copy_file_does_not_exist() {
    let from = Path::new("test/nonexistent-bogus-path");
    let to = Path::new("test/other-bogus-path");

    match fs::copy(&from, &to) {
        Ok(..) => panic!(),
        Err(..) => {
            assert!(!from.exists());
            assert!(!to.exists());
        }
    }
}

#[test]
fn copy_src_does_not_exist() {
    let tmpdir = tmpdir();
    let from = Path::new("test/nonexistent-bogus-path");
    let to = tmpdir.join("out.txt");
    check!(check!(File::create(&to)).write(b"hello"));
    assert!(fs::copy(&from, &to).is_err());
    assert!(!from.exists());
    let mut v = Vec::new();
    check!(check!(File::open(&to)).read_to_end(&mut v));
    assert_eq!(v, b"hello");
}

#[test]
fn copy_file_ok() {
    let tmpdir = tmpdir();
    let input = tmpdir.join("in.txt");
    let out = tmpdir.join("out.txt");

    check!(check!(File::create(&input)).write(b"hello"));
    check!(fs::copy(&input, &out));
    let mut v = Vec::new();
    check!(check!(File::open(&out)).read_to_end(&mut v));
    assert_eq!(v, b"hello");

    assert_eq!(check!(input.metadata()).permissions(), check!(out.metadata()).permissions());
}

#[test]
fn copy_file_dst_dir() {
    let tmpdir = tmpdir();
    let out = tmpdir.join("out");

    check!(File::create(&out));
    match fs::copy(&*out, tmpdir.path()) {
        Ok(..) => panic!(),
        Err(..) => {}
    }
}

#[test]
fn copy_file_dst_exists() {
    let tmpdir = tmpdir();
    let input = tmpdir.join("in");
    let output = tmpdir.join("out");

    check!(check!(File::create(&input)).write("foo".as_bytes()));
    check!(check!(File::create(&output)).write("bar".as_bytes()));
    check!(fs::copy(&input, &output));

    let mut v = Vec::new();
    check!(check!(File::open(&output)).read_to_end(&mut v));
    assert_eq!(v, b"foo".to_vec());
}

#[test]
fn copy_file_src_dir() {
    let tmpdir = tmpdir();
    let out = tmpdir.join("out");

    match fs::copy(tmpdir.path(), &out) {
        Ok(..) => panic!(),
        Err(..) => {}
    }
    assert!(!out.exists());
}

#[test]
fn copy_file_preserves_perm_bits() {
    let tmpdir = tmpdir();
    let input = tmpdir.join("in.txt");
    let out = tmpdir.join("out.txt");

    let attr = check!(check!(File::create(&input)).metadata());
    let mut p = attr.permissions();
    p.set_readonly(true);
    check!(fs::set_permissions(&input, p));
    check!(fs::copy(&input, &out));
    assert!(check!(out.metadata()).permissions().readonly());
    check!(fs::set_permissions(&input, attr.permissions()));
    check!(fs::set_permissions(&out, attr.permissions()));
}

#[test]
#[cfg(windows)]
fn copy_file_preserves_streams() {
    let tmp = tmpdir();
    check!(check!(File::create(tmp.join("in.txt:bunny"))).write("carrot".as_bytes()));
    assert_eq!(check!(fs::copy(tmp.join("in.txt"), tmp.join("out.txt"))), 0);
    assert_eq!(check!(tmp.join("out.txt").metadata()).len(), 0);
    let mut v = Vec::new();
    check!(check!(File::open(tmp.join("out.txt:bunny"))).read_to_end(&mut v));
    assert_eq!(v, b"carrot".to_vec());
}

#[test]
fn copy_file_returns_metadata_len() {
    let tmp = tmpdir();
    let in_path = tmp.join("in.txt");
    let out_path = tmp.join("out.txt");
    check!(check!(File::create(&in_path)).write(b"lettuce"));
    #[cfg(windows)]
    check!(check!(File::create(tmp.join("in.txt:bunny"))).write(b"carrot"));
    let copied_len = check!(fs::copy(&in_path, &out_path));
    assert_eq!(check!(out_path.metadata()).len(), copied_len);
}

#[test]
fn copy_file_follows_dst_symlink() {
    let tmp = tmpdir();
    if !got_symlink_permission(&tmp) {
        return;
    };

    let in_path = tmp.join("in.txt");
    let out_path = tmp.join("out.txt");
    let out_path_symlink = tmp.join("out_symlink.txt");

    check!(fs::write(&in_path, "foo"));
    check!(fs::write(&out_path, "bar"));
    check!(symlink_file(&out_path, &out_path_symlink));

    check!(fs::copy(&in_path, &out_path_symlink));

    assert!(check!(out_path_symlink.symlink_metadata()).file_type().is_symlink());
    assert_eq!(check!(fs::read(&out_path_symlink)), b"foo".to_vec());
    assert_eq!(check!(fs::read(&out_path)), b"foo".to_vec());
}

#[test]
fn symlinks_work() {
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    };

    let input = tmpdir.join("in.txt");
    let out = tmpdir.join("out.txt");

    check!(check!(File::create(&input)).write("foobar".as_bytes()));
    check!(symlink_file(&input, &out));
    assert!(check!(out.symlink_metadata()).file_type().is_symlink());
    assert_eq!(check!(fs::metadata(&out)).len(), check!(fs::metadata(&input)).len());
    let mut v = Vec::new();
    check!(check!(File::open(&out)).read_to_end(&mut v));
    assert_eq!(v, b"foobar".to_vec());
}

#[test]
fn symlink_noexist() {
    // Symlinks can point to things that don't exist
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    };

    // Use a relative path for testing. Symlinks get normalized by Windows,
    // so we might not get the same path back for absolute paths
    check!(symlink_file(&"foo", &tmpdir.join("bar")));
    assert_eq!(check!(fs::read_link(&tmpdir.join("bar"))).to_str().unwrap(), "foo");
}

#[test]
fn read_link() {
    let tmpdir = tmpdir();
    if cfg!(windows) {
        // directory symlink
        assert_eq!(check!(fs::read_link(r"C:\Users\All Users")), Path::new(r"C:\ProgramData"));
        // junction
        assert_eq!(check!(fs::read_link(r"C:\Users\Default User")), Path::new(r"C:\Users\Default"));
        // junction with special permissions
        // Since not all localized windows versions contain the folder "Documents and Settings" in english,
        // we will briefly check, if it exists and otherwise skip the test. Except during CI we will always execute the test.
        if Path::new(r"C:\Documents and Settings\").exists() || env::var_os("CI").is_some() {
            assert_eq!(
                check!(fs::read_link(r"C:\Documents and Settings\")),
                Path::new(r"C:\Users")
            );
        }
        // Check that readlink works with non-drive paths on Windows.
        let link = tmpdir.join("link_unc");
        if got_symlink_permission(&tmpdir) {
            check!(symlink_dir(r"\\localhost\c$\", &link));
            assert_eq!(check!(fs::read_link(&link)), Path::new(r"\\localhost\c$\"));
        };
    }
    let link = tmpdir.join("link");
    if !got_symlink_permission(&tmpdir) {
        return;
    };
    check!(symlink_file(&"foo", &link));
    assert_eq!(check!(fs::read_link(&link)).to_str().unwrap(), "foo");
}

#[test]
fn readlink_not_symlink() {
    let tmpdir = tmpdir();
    match fs::read_link(tmpdir.path()) {
        Ok(..) => panic!("wanted a failure"),
        Err(..) => {}
    }
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating hardlinks
fn links_work() {
    let tmpdir = tmpdir();
    let input = tmpdir.join("in.txt");
    let out = tmpdir.join("out.txt");

    check!(check!(File::create(&input)).write("foobar".as_bytes()));
    check!(fs::hard_link(&input, &out));
    assert_eq!(check!(fs::metadata(&out)).len(), check!(fs::metadata(&input)).len());
    assert_eq!(check!(fs::metadata(&out)).len(), check!(input.metadata()).len());
    let mut v = Vec::new();
    check!(check!(File::open(&out)).read_to_end(&mut v));
    assert_eq!(v, b"foobar".to_vec());

    // can't link to yourself
    match fs::hard_link(&input, &input) {
        Ok(..) => panic!("wanted a failure"),
        Err(..) => {}
    }
    // can't link to something that doesn't exist
    match fs::hard_link(&tmpdir.join("foo"), &tmpdir.join("bar")) {
        Ok(..) => panic!("wanted a failure"),
        Err(..) => {}
    }
}

#[test]
fn chmod_works() {
    let tmpdir = tmpdir();
    let file = tmpdir.join("in.txt");

    check!(File::create(&file));
    let attr = check!(fs::metadata(&file));
    assert!(!attr.permissions().readonly());
    let mut p = attr.permissions();
    p.set_readonly(true);
    check!(fs::set_permissions(&file, p.clone()));
    let attr = check!(fs::metadata(&file));
    assert!(attr.permissions().readonly());

    match fs::set_permissions(&tmpdir.join("foo"), p.clone()) {
        Ok(..) => panic!("wanted an error"),
        Err(..) => {}
    }

    p.set_readonly(false);
    check!(fs::set_permissions(&file, p));
}

#[test]
fn fchmod_works() {
    let tmpdir = tmpdir();
    let path = tmpdir.join("in.txt");

    let file = check!(File::create(&path));
    let attr = check!(fs::metadata(&path));
    assert!(!attr.permissions().readonly());
    let mut p = attr.permissions();
    p.set_readonly(true);
    check!(file.set_permissions(p.clone()));
    let attr = check!(fs::metadata(&path));
    assert!(attr.permissions().readonly());

    p.set_readonly(false);
    check!(file.set_permissions(p));
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
    check!(file.sync_data());
}

#[test]
fn truncate_works() {
    let tmpdir = tmpdir();
    let path = tmpdir.join("in.txt");

    let mut file = check!(File::create(&path));
    check!(file.write(b"foo"));
    check!(file.sync_all());

    // Do some simple things with truncation
    assert_eq!(check!(file.metadata()).len(), 3);
    check!(file.set_len(10));
    assert_eq!(check!(file.metadata()).len(), 10);
    check!(file.write(b"bar"));
    check!(file.sync_all());
    assert_eq!(check!(file.metadata()).len(), 10);

    let mut v = Vec::new();
    check!(check!(File::open(&path)).read_to_end(&mut v));
    assert_eq!(v, b"foobar\0\0\0\0".to_vec());

    // Truncate to a smaller length, don't seek, and then write something.
    // Ensure that the intermediate zeroes are all filled in (we have `seek`ed
    // past the end of the file).
    check!(file.set_len(2));
    assert_eq!(check!(file.metadata()).len(), 2);
    check!(file.write(b"wut"));
    check!(file.sync_all());
    assert_eq!(check!(file.metadata()).len(), 9);
    let mut v = Vec::new();
    check!(check!(File::open(&path)).read_to_end(&mut v));
    assert_eq!(v, b"fo\0\0\0\0wut".to_vec());
}

#[test]
fn open_flavors() {
    use crate::fs::OpenOptions as OO;
    fn c<T: Clone>(t: &T) -> T {
        t.clone()
    }

    let tmpdir = tmpdir();

    let mut r = OO::new();
    r.read(true);
    let mut w = OO::new();
    w.write(true);
    let mut rw = OO::new();
    rw.read(true).write(true);
    let mut a = OO::new();
    a.append(true);
    let mut ra = OO::new();
    ra.read(true).append(true);

    #[cfg(windows)]
    let invalid_options = 87; // ERROR_INVALID_PARAMETER
    #[cfg(all(unix, not(target_os = "vxworks")))]
    let invalid_options = "Invalid argument";
    #[cfg(target_os = "vxworks")]
    let invalid_options = "invalid argument";

    // Test various combinations of creation modes and access modes.
    //
    // Allowed:
    // creation mode           | read  | write | read-write | append | read-append |
    // :-----------------------|:-----:|:-----:|:----------:|:------:|:-----------:|
    // not set (open existing) |   X   |   X   |     X      |   X    |      X      |
    // create                  |       |   X   |     X      |   X    |      X      |
    // truncate                |       |   X   |     X      |        |             |
    // create and truncate     |       |   X   |     X      |        |             |
    // create_new              |       |   X   |     X      |   X    |      X      |
    //
    // tested in reverse order, so 'create_new' creates the file, and 'open existing' opens it.

    // write-only
    check!(c(&w).create_new(true).open(&tmpdir.join("a")));
    check!(c(&w).create(true).truncate(true).open(&tmpdir.join("a")));
    check!(c(&w).truncate(true).open(&tmpdir.join("a")));
    check!(c(&w).create(true).open(&tmpdir.join("a")));
    check!(c(&w).open(&tmpdir.join("a")));

    // read-only
    error!(c(&r).create_new(true).open(&tmpdir.join("b")), invalid_options);
    error!(c(&r).create(true).truncate(true).open(&tmpdir.join("b")), invalid_options);
    error!(c(&r).truncate(true).open(&tmpdir.join("b")), invalid_options);
    error!(c(&r).create(true).open(&tmpdir.join("b")), invalid_options);
    check!(c(&r).open(&tmpdir.join("a"))); // try opening the file created with write_only

    // read-write
    check!(c(&rw).create_new(true).open(&tmpdir.join("c")));
    check!(c(&rw).create(true).truncate(true).open(&tmpdir.join("c")));
    check!(c(&rw).truncate(true).open(&tmpdir.join("c")));
    check!(c(&rw).create(true).open(&tmpdir.join("c")));
    check!(c(&rw).open(&tmpdir.join("c")));

    // append
    check!(c(&a).create_new(true).open(&tmpdir.join("d")));
    error!(c(&a).create(true).truncate(true).open(&tmpdir.join("d")), invalid_options);
    error!(c(&a).truncate(true).open(&tmpdir.join("d")), invalid_options);
    check!(c(&a).create(true).open(&tmpdir.join("d")));
    check!(c(&a).open(&tmpdir.join("d")));

    // read-append
    check!(c(&ra).create_new(true).open(&tmpdir.join("e")));
    error!(c(&ra).create(true).truncate(true).open(&tmpdir.join("e")), invalid_options);
    error!(c(&ra).truncate(true).open(&tmpdir.join("e")), invalid_options);
    check!(c(&ra).create(true).open(&tmpdir.join("e")));
    check!(c(&ra).open(&tmpdir.join("e")));

    // Test opening a file without setting an access mode
    let mut blank = OO::new();
    error!(blank.create(true).open(&tmpdir.join("f")), invalid_options);

    // Test write works
    check!(check!(File::create(&tmpdir.join("h"))).write("foobar".as_bytes()));

    // Test write fails for read-only
    check!(r.open(&tmpdir.join("h")));
    {
        let mut f = check!(r.open(&tmpdir.join("h")));
        assert!(f.write("wut".as_bytes()).is_err());
    }

    // Test write overwrites
    {
        let mut f = check!(c(&w).open(&tmpdir.join("h")));
        check!(f.write("baz".as_bytes()));
    }
    {
        let mut f = check!(c(&r).open(&tmpdir.join("h")));
        let mut b = vec![0; 6];
        check!(f.read(&mut b));
        assert_eq!(b, "bazbar".as_bytes());
    }

    // Test truncate works
    {
        let mut f = check!(c(&w).truncate(true).open(&tmpdir.join("h")));
        check!(f.write("foo".as_bytes()));
    }
    assert_eq!(check!(fs::metadata(&tmpdir.join("h"))).len(), 3);

    // Test append works
    assert_eq!(check!(fs::metadata(&tmpdir.join("h"))).len(), 3);
    {
        let mut f = check!(c(&a).open(&tmpdir.join("h")));
        check!(f.write("bar".as_bytes()));
    }
    assert_eq!(check!(fs::metadata(&tmpdir.join("h"))).len(), 6);

    // Test .append(true) equals .write(true).append(true)
    {
        let mut f = check!(c(&w).append(true).open(&tmpdir.join("h")));
        check!(f.write("baz".as_bytes()));
    }
    assert_eq!(check!(fs::metadata(&tmpdir.join("h"))).len(), 9);
}

#[test]
fn _assert_send_sync() {
    fn _assert_send_sync<T: Send + Sync>() {}
    _assert_send_sync::<OpenOptions>();
}

#[test]
fn binary_file() {
    let mut bytes = [0; 1024];
    crate::test_helpers::test_rng().fill_bytes(&mut bytes);

    let tmpdir = tmpdir();

    check!(check!(File::create(&tmpdir.join("test"))).write(&bytes));
    let mut v = Vec::new();
    check!(check!(File::open(&tmpdir.join("test"))).read_to_end(&mut v));
    assert!(v == &bytes[..]);
}

#[test]
fn write_then_read() {
    let mut bytes = [0; 1024];
    crate::test_helpers::test_rng().fill_bytes(&mut bytes);

    let tmpdir = tmpdir();

    check!(fs::write(&tmpdir.join("test"), &bytes[..]));
    let v = check!(fs::read(&tmpdir.join("test")));
    assert!(v == &bytes[..]);

    check!(fs::write(&tmpdir.join("not-utf8"), &[0xFF]));
    error_contains!(
        fs::read_to_string(&tmpdir.join("not-utf8")),
        "stream did not contain valid UTF-8"
    );

    let s = "𐁁𐀓𐀠𐀴𐀍";
    check!(fs::write(&tmpdir.join("utf8"), s.as_bytes()));
    let string = check!(fs::read_to_string(&tmpdir.join("utf8")));
    assert_eq!(string, s);
}

#[test]
fn file_try_clone() {
    let tmpdir = tmpdir();

    let mut f1 =
        check!(OpenOptions::new().read(true).write(true).create(true).open(&tmpdir.join("test")));
    let mut f2 = check!(f1.try_clone());

    check!(f1.write_all(b"hello world"));
    check!(f1.seek(SeekFrom::Start(2)));

    let mut buf = vec![];
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
    let mut perm = check!(fs::metadata(&path)).permissions();
    perm.set_readonly(true);
    check!(fs::set_permissions(&path, perm));
    check!(fs::remove_file(&path));
}

#[test]
fn mkdir_trailing_slash() {
    let tmpdir = tmpdir();
    let path = tmpdir.join("file");
    check!(fs::create_dir_all(&path.join("a/")));
}

#[test]
fn canonicalize_works_simple() {
    let tmpdir = tmpdir();
    let tmpdir = fs::canonicalize(tmpdir.path()).unwrap();
    let file = tmpdir.join("test");
    File::create(&file).unwrap();
    assert_eq!(fs::canonicalize(&file).unwrap(), file);
}

#[test]
fn realpath_works() {
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    };

    let tmpdir = fs::canonicalize(tmpdir.path()).unwrap();
    let file = tmpdir.join("test");
    let dir = tmpdir.join("test2");
    let link = dir.join("link");
    let linkdir = tmpdir.join("test3");

    File::create(&file).unwrap();
    fs::create_dir(&dir).unwrap();
    symlink_file(&file, &link).unwrap();
    symlink_dir(&dir, &linkdir).unwrap();

    assert!(link.symlink_metadata().unwrap().file_type().is_symlink());

    assert_eq!(fs::canonicalize(&tmpdir).unwrap(), tmpdir);
    assert_eq!(fs::canonicalize(&file).unwrap(), file);
    assert_eq!(fs::canonicalize(&link).unwrap(), file);
    assert_eq!(fs::canonicalize(&linkdir).unwrap(), dir);
    assert_eq!(fs::canonicalize(&linkdir.join("link")).unwrap(), file);
}

#[test]
fn realpath_works_tricky() {
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    };

    let tmpdir = fs::canonicalize(tmpdir.path()).unwrap();
    let a = tmpdir.join("a");
    let b = a.join("b");
    let c = b.join("c");
    let d = a.join("d");
    let e = d.join("e");
    let f = a.join("f");

    fs::create_dir_all(&b).unwrap();
    fs::create_dir_all(&d).unwrap();
    File::create(&f).unwrap();
    if cfg!(not(windows)) {
        symlink_file("../d/e", &c).unwrap();
        symlink_file("../f", &e).unwrap();
    }
    if cfg!(windows) {
        symlink_file(r"..\d\e", &c).unwrap();
        symlink_file(r"..\f", &e).unwrap();
    }

    assert_eq!(fs::canonicalize(&c).unwrap(), f);
    assert_eq!(fs::canonicalize(&e).unwrap(), f);
}

#[test]
fn dir_entry_methods() {
    let tmpdir = tmpdir();

    fs::create_dir_all(&tmpdir.join("a")).unwrap();
    File::create(&tmpdir.join("b")).unwrap();

    for file in tmpdir.path().read_dir().unwrap().map(|f| f.unwrap()) {
        let fname = file.file_name();
        match fname.to_str() {
            Some("a") => {
                assert!(file.file_type().unwrap().is_dir());
                assert!(file.metadata().unwrap().is_dir());
            }
            Some("b") => {
                assert!(file.file_type().unwrap().is_file());
                assert!(file.metadata().unwrap().is_file());
            }
            f => panic!("unknown file name: {f:?}"),
        }
    }
}

#[test]
fn dir_entry_debug() {
    let tmpdir = tmpdir();
    File::create(&tmpdir.join("b")).unwrap();
    let mut read_dir = tmpdir.path().read_dir().unwrap();
    let dir_entry = read_dir.next().unwrap().unwrap();
    let actual = format!("{dir_entry:?}");
    let expected = format!("DirEntry({:?})", dir_entry.0.path());
    assert_eq!(actual, expected);
}

#[test]
fn read_dir_not_found() {
    let res = fs::read_dir("/path/that/does/not/exist");
    assert_eq!(res.err().unwrap().kind(), ErrorKind::NotFound);
}

#[test]
fn file_open_not_found() {
    let res = File::open("/path/that/does/not/exist");
    assert_eq!(res.err().unwrap().kind(), ErrorKind::NotFound);
}

#[test]
fn create_dir_all_with_junctions() {
    let tmpdir = tmpdir();
    let target = tmpdir.join("target");

    let junction = tmpdir.join("junction");
    let b = junction.join("a/b");

    let link = tmpdir.join("link");
    let d = link.join("c/d");

    fs::create_dir(&target).unwrap();

    check!(junction_point(&target, &junction));
    check!(fs::create_dir_all(&b));
    // the junction itself is not a directory, but `is_dir()` on a Path
    // follows links
    assert!(junction.is_dir());
    assert!(b.exists());

    if !got_symlink_permission(&tmpdir) {
        return;
    };
    check!(symlink_dir(&target, &link));
    check!(fs::create_dir_all(&d));
    assert!(link.is_dir());
    assert!(d.exists());
}

#[test]
fn metadata_access_times() {
    let tmpdir = tmpdir();

    let b = tmpdir.join("b");
    File::create(&b).unwrap();

    let a = check!(fs::metadata(&tmpdir.path()));
    let b = check!(fs::metadata(&b));

    assert_eq!(check!(a.accessed()), check!(a.accessed()));
    assert_eq!(check!(a.modified()), check!(a.modified()));
    assert_eq!(check!(b.accessed()), check!(b.modified()));

    if cfg!(target_vendor = "apple") || cfg!(target_os = "windows") {
        check!(a.created());
        check!(b.created());
    }

    if cfg!(target_os = "linux") {
        // Not always available
        match (a.created(), b.created()) {
            (Ok(t1), Ok(t2)) => assert!(t1 <= t2),
            (Err(e1), Err(e2))
                if e1.kind() == ErrorKind::Uncategorized
                    && e2.kind() == ErrorKind::Uncategorized
                    || e1.kind() == ErrorKind::Unsupported
                        && e2.kind() == ErrorKind::Unsupported => {}
            (a, b) => {
                panic!("creation time must be always supported or not supported: {a:?} {b:?}")
            }
        }
    }
}

/// Test creating hard links to symlinks.
#[test]
#[cfg_attr(target_os = "android", ignore)] // Android SELinux rules prevent creating hardlinks
fn symlink_hard_link() {
    let tmpdir = tmpdir();
    if !got_symlink_permission(&tmpdir) {
        return;
    };

    // Create "file", a file.
    check!(fs::File::create(tmpdir.join("file")));

    // Create "symlink", a symlink to "file".
    check!(symlink_file("file", tmpdir.join("symlink")));

    // Create "hard_link", a hard link to "symlink".
    check!(fs::hard_link(tmpdir.join("symlink"), tmpdir.join("hard_link")));

    // "hard_link" should appear as a symlink.
    assert!(check!(fs::symlink_metadata(tmpdir.join("hard_link"))).file_type().is_symlink());

    // We should be able to open "file" via any of the above names.
    let _ = check!(fs::File::open(tmpdir.join("file")));
    assert!(fs::File::open(tmpdir.join("file.renamed")).is_err());
    let _ = check!(fs::File::open(tmpdir.join("symlink")));
    let _ = check!(fs::File::open(tmpdir.join("hard_link")));

    // Rename "file" to "file.renamed".
    check!(fs::rename(tmpdir.join("file"), tmpdir.join("file.renamed")));

    // Now, the symlink and the hard link should be dangling.
    assert!(fs::File::open(tmpdir.join("file")).is_err());
    let _ = check!(fs::File::open(tmpdir.join("file.renamed")));
    assert!(fs::File::open(tmpdir.join("symlink")).is_err());
    assert!(fs::File::open(tmpdir.join("hard_link")).is_err());

    // The symlink and the hard link should both still point to "file".
    assert!(fs::read_link(tmpdir.join("file")).is_err());
    assert!(fs::read_link(tmpdir.join("file.renamed")).is_err());
    assert_eq!(check!(fs::read_link(tmpdir.join("symlink"))), Path::new("file"));
    assert_eq!(check!(fs::read_link(tmpdir.join("hard_link"))), Path::new("file"));

    // Remove "file.renamed".
    check!(fs::remove_file(tmpdir.join("file.renamed")));

    // Now, we can't open the file by any name.
    assert!(fs::File::open(tmpdir.join("file")).is_err());
    assert!(fs::File::open(tmpdir.join("file.renamed")).is_err());
    assert!(fs::File::open(tmpdir.join("symlink")).is_err());
    assert!(fs::File::open(tmpdir.join("hard_link")).is_err());

    // "hard_link" should still appear as a symlink.
    assert!(check!(fs::symlink_metadata(tmpdir.join("hard_link"))).file_type().is_symlink());
}

/// Ensure `fs::create_dir` works on Windows with longer paths.
#[test]
#[cfg(windows)]
fn create_dir_long_paths() {
    use crate::ffi::OsStr;
    use crate::iter;
    use crate::os::windows::ffi::OsStrExt;
    const PATH_LEN: usize = 247;

    let tmpdir = tmpdir();
    let mut path = tmpdir.path().to_path_buf();
    path.push("a");
    let mut path = path.into_os_string();

    let utf16_len = path.encode_wide().count();
    if utf16_len >= PATH_LEN {
        // Skip the test in the unlikely event the local user has a long temp directory path.
        // This should not affect CI.
        return;
    }
    // Increase the length of the path.
    path.extend(iter::repeat(OsStr::new("a")).take(PATH_LEN - utf16_len));

    // This should succeed.
    fs::create_dir(&path).unwrap();

    // This will fail if the path isn't converted to verbatim.
    path.push("a");
    fs::create_dir(&path).unwrap();

    // #90940: Ensure an empty path returns the "Not Found" error.
    let path = Path::new("");
    assert_eq!(path.canonicalize().unwrap_err().kind(), crate::io::ErrorKind::NotFound);
}

/// Ensure ReadDir works on large directories.
/// Regression test for https://github.com/rust-lang/rust/issues/93384.
#[test]
fn read_large_dir() {
    let tmpdir = tmpdir();

    let count = 32 * 1024;
    for i in 0..count {
        check!(fs::File::create(tmpdir.join(&i.to_string())));
    }

    for entry in fs::read_dir(tmpdir.path()).unwrap() {
        entry.unwrap();
    }
}

/// Test the fallback for getting the metadata of files like hiberfil.sys that
/// Windows holds a special lock on, preventing normal means of querying
/// metadata. See #96980.
///
/// Note this fails in CI because `hiberfil.sys` does not actually exist there.
/// Therefore it's marked as ignored.
#[test]
#[ignore]
#[cfg(windows)]
fn hiberfil_sys() {
    let hiberfil = Path::new(r"C:\hiberfil.sys");
    assert_eq!(true, hiberfil.try_exists().unwrap());
    fs::symlink_metadata(hiberfil).unwrap();
    fs::metadata(hiberfil).unwrap();
    assert_eq!(true, hiberfil.exists());
}

/// Test that two different ways of obtaining the FileType give the same result.
/// Cf. https://github.com/rust-lang/rust/issues/104900
#[test]
fn test_eq_direntry_metadata() {
    let tmpdir = tmpdir();
    let file_path = tmpdir.join("file");
    File::create(file_path).unwrap();
    for e in fs::read_dir(tmpdir.path()).unwrap() {
        let e = e.unwrap();
        let p = e.path();
        let ft1 = e.file_type().unwrap();
        let ft2 = p.metadata().unwrap().file_type();
        assert_eq!(ft1, ft2);
    }
}

/// Regression test for https://github.com/rust-lang/rust/issues/50619.
#[test]
#[cfg(target_os = "linux")]
fn test_read_dir_infinite_loop() {
    use crate::io::ErrorKind;
    use crate::process::Command;

    // Create a zombie child process
    let Ok(mut child) = Command::new("echo").spawn() else { return };

    // Make sure the process is (un)dead
    match child.kill() {
        // InvalidInput means the child already exited
        Err(e) if e.kind() != ErrorKind::InvalidInput => return,
        _ => {}
    }

    // open() on this path will succeed, but readdir() will fail
    let id = child.id();
    let path = format!("/proc/{id}/net");

    // Skip the test if we can't open the directory in the first place
    let Ok(dir) = fs::read_dir(path) else { return };

    // Check for duplicate errors
    assert!(dir.filter(|e| e.is_err()).take(2).count() < 2);
}

#[test]
fn rename_directory() {
    let tmpdir = tmpdir();
    let old_path = tmpdir.join("foo/bar/baz");
    fs::create_dir_all(&old_path).unwrap();
    let test_file = &old_path.join("temp.txt");

    File::create(test_file).unwrap();

    let new_path = tmpdir.join("quux/blat");
    fs::create_dir_all(&new_path).unwrap();
    fs::rename(&old_path, &new_path.join("newdir")).unwrap();
    assert!(new_path.join("newdir").is_dir());
    assert!(new_path.join("newdir/temp.txt").exists());
}

#[test]
fn test_file_times() {
    #[cfg(target_vendor = "apple")]
    use crate::os::darwin::fs::FileTimesExt;
    #[cfg(windows)]
    use crate::os::windows::fs::FileTimesExt;

    let tmp = tmpdir();
    let file = File::create(tmp.join("foo")).unwrap();
    let mut times = FileTimes::new();
    let accessed = SystemTime::UNIX_EPOCH + Duration::from_secs(12345);
    let modified = SystemTime::UNIX_EPOCH + Duration::from_secs(54321);
    times = times.set_accessed(accessed).set_modified(modified);
    #[cfg(any(windows, target_vendor = "apple"))]
    let created = SystemTime::UNIX_EPOCH + Duration::from_secs(32123);
    #[cfg(any(windows, target_vendor = "apple"))]
    {
        times = times.set_created(created);
    }
    match file.set_times(times) {
        // Allow unsupported errors on platforms which don't support setting times.
        #[cfg(not(any(
            windows,
            all(
                unix,
                not(any(
                    target_os = "android",
                    target_os = "redox",
                    target_os = "espidf",
                    target_os = "horizon"
                ))
            )
        )))]
        Err(e) if e.kind() == ErrorKind::Unsupported => return,
        Err(e) => panic!("error setting file times: {e:?}"),
        Ok(_) => {}
    }
    let metadata = file.metadata().unwrap();
    assert_eq!(metadata.accessed().unwrap(), accessed);
    assert_eq!(metadata.modified().unwrap(), modified);
    #[cfg(any(windows, target_vendor = "apple"))]
    {
        assert_eq!(metadata.created().unwrap(), created);
    }
}

#[test]
#[cfg(target_vendor = "apple")]
fn test_file_times_pre_epoch_with_nanos() {
    use crate::os::darwin::fs::FileTimesExt;

    let tmp = tmpdir();
    let file = File::create(tmp.join("foo")).unwrap();

    for (accessed, modified, created) in [
        // The first round is to set filetimes to something we know works, but this time
        // it's validated with nanoseconds as well which probe the numeric boundary.
        (
            SystemTime::UNIX_EPOCH + Duration::new(12345, 1),
            SystemTime::UNIX_EPOCH + Duration::new(54321, 100_000_000),
            SystemTime::UNIX_EPOCH + Duration::new(32123, 999_999_999),
        ),
        // The second rounds uses pre-epoch dates along with nanoseconds that probe
        // the numeric boundary.
        (
            SystemTime::UNIX_EPOCH - Duration::new(1, 1),
            SystemTime::UNIX_EPOCH - Duration::new(60, 100_000_000),
            SystemTime::UNIX_EPOCH - Duration::new(3600, 999_999_999),
        ),
    ] {
        let mut times = FileTimes::new();
        times = times.set_accessed(accessed).set_modified(modified).set_created(created);
        file.set_times(times).unwrap();

        let metadata = file.metadata().unwrap();
        assert_eq!(metadata.accessed().unwrap(), accessed);
        assert_eq!(metadata.modified().unwrap(), modified);
        assert_eq!(metadata.created().unwrap(), created);
    }
}

#[test]
#[cfg(windows)]
fn windows_unix_socket_exists() {
    use crate::sys::{c, net};
    use crate::{mem, ptr};

    let tmp = tmpdir();
    let socket_path = tmp.join("socket");

    // std doesn't currently support Unix sockets on Windows so manually create one here.
    net::init();
    unsafe {
        let socket = c::WSASocketW(
            c::AF_UNIX as i32,
            c::SOCK_STREAM,
            0,
            ptr::null_mut(),
            0,
            c::WSA_FLAG_OVERLAPPED | c::WSA_FLAG_NO_HANDLE_INHERIT,
        );
        // AF_UNIX is not supported on earlier versions of Windows,
        // so skip this test if it's unsupported and we're not in CI.
        if socket == c::INVALID_SOCKET {
            let error = c::WSAGetLastError();
            if env::var_os("CI").is_none() && error == c::WSAEAFNOSUPPORT {
                return;
            } else {
                panic!("Creating AF_UNIX socket failed (OS error {error})");
            }
        }
        let mut addr = c::SOCKADDR_UN { sun_family: c::AF_UNIX, sun_path: mem::zeroed() };
        let bytes = socket_path.as_os_str().as_encoded_bytes();
        let bytes = core::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len());
        addr.sun_path[..bytes.len()].copy_from_slice(bytes);
        let len = mem::size_of_val(&addr) as i32;
        let result = c::bind(socket, (&raw const addr).cast::<c::SOCKADDR>(), len);
        c::closesocket(socket);
        assert_eq!(result, 0);
    }
    // Make sure all ways of testing a file exist work for a Unix socket.
    assert_eq!(socket_path.exists(), true);
    assert_eq!(socket_path.try_exists().unwrap(), true);
    assert_eq!(socket_path.metadata().is_ok(), true);
}

#[cfg(windows)]
#[test]
fn test_hidden_file_truncation() {
    // Make sure that File::create works on an existing hidden file. See #115745.
    let tmpdir = tmpdir();
    let path = tmpdir.join("hidden_file.txt");

    // Create a hidden file.
    const FILE_ATTRIBUTE_HIDDEN: u32 = 2;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .attributes(FILE_ATTRIBUTE_HIDDEN)
        .open(&path)
        .unwrap();
    file.write("hidden world!".as_bytes()).unwrap();
    file.flush().unwrap();
    drop(file);

    // Create a new file by truncating the existing one.
    let file = File::create(&path).unwrap();
    let metadata = file.metadata().unwrap();
    assert_eq!(metadata.len(), 0);
}

#[cfg(windows)]
#[test]
fn test_rename_file_over_open_file() {
    // Make sure that std::fs::rename works if the target file is already opened with FILE_SHARE_DELETE. See #123985.
    let tmpdir = tmpdir();

    // Create source with test data to read.
    let source_path = tmpdir.join("source_file.txt");
    fs::write(&source_path, b"source hello world").unwrap();

    // Create target file with test data to read;
    let target_path = tmpdir.join("target_file.txt");
    fs::write(&target_path, b"target hello world").unwrap();

    // Open target file
    let target_file = fs::File::open(&target_path).unwrap();

    // Rename source
    fs::rename(source_path, &target_path).unwrap();

    core::mem::drop(target_file);
    assert_eq!(fs::read(target_path).unwrap(), b"source hello world");
}

#[test]
#[cfg(windows)]
fn test_rename_directory_to_non_empty_directory() {
    // Renaming a directory over a non-empty existing directory should fail on Windows.
    let tmpdir: TempDir = tmpdir();

    let source_path = tmpdir.join("source_directory");
    let target_path = tmpdir.join("target_directory");

    fs::create_dir(&source_path).unwrap();
    fs::create_dir(&target_path).unwrap();

    fs::write(target_path.join("target_file.txt"), b"target hello world").unwrap();

    error!(fs::rename(source_path, target_path), 145); // ERROR_DIR_NOT_EMPTY
}

#[test]
fn test_rename_symlink() {
    let tmpdir = tmpdir();
    let original = tmpdir.join("original");
    let dest = tmpdir.join("dest");
    let not_exist = Path::new("does not exist");

    symlink_file(not_exist, &original).unwrap();
    fs::rename(&original, &dest).unwrap();
    // Make sure that renaming `original` to `dest` preserves the symlink.
    assert_eq!(fs::read_link(&dest).unwrap().as_path(), not_exist);
}

#[test]
#[cfg(windows)]
fn test_rename_junction() {
    let tmpdir = tmpdir();
    let original = tmpdir.join("original");
    let dest = tmpdir.join("dest");
    let not_exist = Path::new("does not exist");

    junction_point(&not_exist, &original).unwrap();
    fs::rename(&original, &dest).unwrap();

    // Make sure that renaming `original` to `dest` preserves the junction point.
    // Junction links are always absolute so we just check the file name is correct.
    assert_eq!(fs::read_link(&dest).unwrap().file_name(), Some(not_exist.as_os_str()));
}
