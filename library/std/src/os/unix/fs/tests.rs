use super::*;

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

    let filename = dir.join("preadv.txt");
    {
        let mut file = fs::File::create(&filename).unwrap();
        file.write_all(msg).unwrap();
    }
    let expected = {
        let file = fs::File::options().write(true).open(&filename).unwrap();
        let buf0 = b"    ";
        let buf1 = b"great  ";

        let iovec = [io::IoSlice::new(buf0), io::IoSlice::new(buf1)];

        let n = file.write_vectored_at(&iovec, 11).unwrap();

        assert!(n == 4 || n == 11);

        if n == 4 { b"pwritev is     working!" } else { b"pwritev is     great  !" }
    };

    let content = fs::read(&filename).unwrap();
    assert_eq!(&content, expected);
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
