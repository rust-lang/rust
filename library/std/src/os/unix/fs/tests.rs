use core::assert_matches;

use super::*;
use crate::byte_str::ByteStr;
use crate::io::Seek;
use crate::test_helpers::tmpdir;

#[test]
fn read_vectored_at() {
    let msg = b"preadv is working!";
    let dir = crate::test_helpers::tmpdir();

    let filename = dir.join("preadv.txt");
    {
        let mut file = fs::File::create(&filename).unwrap();
        file.write_all(msg).unwrap();
    }
    {
        let file = fs::File::open(&filename).unwrap();
        let mut buf0 = [0; 4];
        let mut buf1 = [0; 3];

        let mut iovec = [io::IoSliceMut::new(&mut buf0), io::IoSliceMut::new(&mut buf1)];

        let n = file.read_vectored_at(&mut iovec, 4).unwrap();

        assert!(n == 4 || n == 7);
        assert_eq!(&buf0, b"dv i");

        if n == 7 {
            assert_eq!(&buf1, b"s w");
        }
    }
}

#[test]
fn write_vectored_at() {
    let msg = b"pwritev is not working!";
    let dir = crate::test_helpers::tmpdir();

    let use_append = write_at_append_should_work();

    let filename = dir.join("preadv.txt");
    {
        let mut file = fs::File::create(&filename).unwrap();
        file.write_all(msg).unwrap();
    }
    let expected = {
        // Open in append mode to test that positioned writes bypass O_APPEND.
        let mut file =
            fs::File::options().write(!use_append).append(use_append).open(&filename).unwrap();
        let buf0 = b"    ";
        let buf1 = b"great  ";

        let iovec = [io::IoSlice::new(buf0), io::IoSlice::new(buf1)];

        assert_eq!(file.stream_position().unwrap(), 0);
        let n = file.write_vectored_at(&iovec, 11).unwrap();
        assert_eq!(file.stream_position().unwrap(), 0);

        assert!(n == 4 || n == 11);

        if n == 4 { b"pwritev is     working!" } else { b"pwritev is     great  !" }
    };

    let content = fs::read(&filename).unwrap();
    assert_eq!(ByteStr::new(&content), ByteStr::new(expected));
}

#[test]
fn file_test_append_write_at() {
    use crate::os::unix::fs::FileExt;

    let use_append = write_at_append_should_work();

    let tmpdir = tmpdir();
    let filename = tmpdir.join("file_test_append_write_at.txt");
    let msg = b"it's not working!";
    assert_matches!(fs::write(&filename, &msg), Ok(()));
    let mut f = fs::File::options().write(!use_append).append(use_append).open(&filename).unwrap();
    assert_matches!(f.stream_position(), Ok(0));
    assert_matches!(f.write_at(b"   ", 5), Ok(3));
    assert_matches!(f.stream_position(), Ok(0));

    let content = fs::read(&filename).unwrap();
    assert_eq!(ByteStr::new(&content), ByteStr::new(b"it's     working!"));
}

#[test]
fn read_buf_exact_at_interrupted() {
    struct InterruptOnce(crate::cell::Cell<bool>);

    impl FileExt for InterruptOnce {
        fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
            if self.0.replace(false) {
                return Err(io::ErrorKind::Interrupted.into());
            }
            let data = &b"pread is working!"[offset as usize..];
            let n = buf.len().min(data.len());
            buf[..n].copy_from_slice(&data[..n]);
            Ok(n)
        }

        fn write_at(&self, _buf: &[u8], _offset: u64) -> io::Result<usize> {
            unimplemented!()
        }
    }

    let file = InterruptOnce(crate::cell::Cell::new(true));
    let mut buf = [0; 10];
    let mut buf = io::BorrowedBuf::from(buf.as_mut_slice());
    file.read_buf_exact_at(buf.unfilled(), 6).unwrap();
    assert_eq!(buf.filled(), b"is working");
}

#[test]
fn test_mkfifo() {
    let tmp_dir = crate::test_helpers::tmpdir();

    let fifo = tmp_dir.path().join("fifo");

    mkfifo(&fifo, Permissions::from_mode(0o774)).unwrap();

    let mut wx = fs::File::options().read(true).write(true).open(&fifo).unwrap();
    let mut rx = fs::File::open(fifo).unwrap();

    wx.write_all(b"hello, world!").unwrap();
    drop(wx);

    let mut s = String::new();
    rx.read_to_string(&mut s).unwrap();

    assert_eq!(s, "hello, world!");
}

/// Returns true if write_at is expected to work on O_APPEND files.
fn write_at_append_should_work() -> bool {
    #[cfg(target_os = "linux")]
    unsafe {
        use crate::ffi::CStr;
        use crate::mem;

        // SAFETY: the struct is all chars, so zeroing is fine.
        let mut buf: libc::utsname = mem::zeroed();
        if libc::uname(&mut buf) != 0 {
            panic!("uname failed");
        }

        // SAFETY: uname promises to return null-terminated strings
        let release = CStr::from_ptr(buf.release.as_ptr()).to_str().unwrap_or("");

        // RWF_NOAPPEND has been observed to not work properly on GHA runners
        // at least under  ubuntu 24.04 and 26.04.
        // Perhaps some virtualization bug.
        if release.contains("azure") {
            return false;
        }

        let mut parts = release.split(".");
        let major = parts.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
        let minor = parts.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);

        // RWF_NOAPPEND was introduced in 6.9
        if (major, minor) >= (6, 9) {
            return true;
        }
    }

    true
}
