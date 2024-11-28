use super::{BorrowedBuf, Cursor, SeekFrom, repeat};
use crate::cmp::{self, min};
use crate::io::{
    self, BufRead, BufReader, DEFAULT_BUF_SIZE, IoSlice, IoSliceMut, Read, Seek, Write,
};
use crate::mem::MaybeUninit;
use crate::ops::Deref;

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn read_until() {
    let mut buf = Cursor::new(&b"12"[..]);
    let mut v = Vec::new();
    assert_eq!(buf.read_until(b'3', &mut v).unwrap(), 2);
    assert_eq!(v, b"12");

    let mut buf = Cursor::new(&b"1233"[..]);
    let mut v = Vec::new();
    assert_eq!(buf.read_until(b'3', &mut v).unwrap(), 3);
    assert_eq!(v, b"123");
    v.truncate(0);
    assert_eq!(buf.read_until(b'3', &mut v).unwrap(), 1);
    assert_eq!(v, b"3");
    v.truncate(0);
    assert_eq!(buf.read_until(b'3', &mut v).unwrap(), 0);
    assert_eq!(v, []);
}

#[test]
fn skip_until() {
    let bytes: &[u8] = b"read\0ignore\0read\0ignore\0read\0ignore\0";
    let mut reader = BufReader::new(bytes);

    // read from the bytes, alternating between
    // consuming `read\0`s and skipping `ignore\0`s
    loop {
        // consume `read\0`
        let mut out = Vec::new();
        let read = reader.read_until(0, &mut out).unwrap();
        if read == 0 {
            // eof
            break;
        } else {
            assert_eq!(out, b"read\0");
            assert_eq!(read, b"read\0".len());
        }

        // skip past `ignore\0`
        let skipped = reader.skip_until(0).unwrap();
        assert_eq!(skipped, b"ignore\0".len());
    }

    // ensure we are at the end of the byte slice and that we can skip no further
    // also ensure skip_until matches the behavior of read_until at EOF
    let skipped = reader.skip_until(0).unwrap();
    assert_eq!(skipped, 0);
}

#[test]
fn split() {
    let buf = Cursor::new(&b"12"[..]);
    let mut s = buf.split(b'3');
    assert_eq!(s.next().unwrap().unwrap(), vec![b'1', b'2']);
    assert!(s.next().is_none());

    let buf = Cursor::new(&b"1233"[..]);
    let mut s = buf.split(b'3');
    assert_eq!(s.next().unwrap().unwrap(), vec![b'1', b'2']);
    assert_eq!(s.next().unwrap().unwrap(), vec![]);
    assert!(s.next().is_none());
}

#[test]
fn read_line() {
    let mut buf = Cursor::new(&b"12"[..]);
    let mut v = String::new();
    assert_eq!(buf.read_line(&mut v).unwrap(), 2);
    assert_eq!(v, "12");

    let mut buf = Cursor::new(&b"12\n\n"[..]);
    let mut v = String::new();
    assert_eq!(buf.read_line(&mut v).unwrap(), 3);
    assert_eq!(v, "12\n");
    v.truncate(0);
    assert_eq!(buf.read_line(&mut v).unwrap(), 1);
    assert_eq!(v, "\n");
    v.truncate(0);
    assert_eq!(buf.read_line(&mut v).unwrap(), 0);
    assert_eq!(v, "");
}

#[test]
fn lines() {
    let buf = Cursor::new(&b"12\r"[..]);
    let mut s = buf.lines();
    assert_eq!(s.next().unwrap().unwrap(), "12\r".to_string());
    assert!(s.next().is_none());

    let buf = Cursor::new(&b"12\r\n\n"[..]);
    let mut s = buf.lines();
    assert_eq!(s.next().unwrap().unwrap(), "12".to_string());
    assert_eq!(s.next().unwrap().unwrap(), "".to_string());
    assert!(s.next().is_none());
}

#[test]
fn buf_read_has_data_left() {
    let mut buf = Cursor::new(&b"abcd"[..]);
    assert!(buf.has_data_left().unwrap());
    buf.read_exact(&mut [0; 2]).unwrap();
    assert!(buf.has_data_left().unwrap());
    buf.read_exact(&mut [0; 2]).unwrap();
    assert!(!buf.has_data_left().unwrap());
}

#[test]
fn read_to_end() {
    let mut c = Cursor::new(&b""[..]);
    let mut v = Vec::new();
    assert_eq!(c.read_to_end(&mut v).unwrap(), 0);
    assert_eq!(v, []);

    let mut c = Cursor::new(&b"1"[..]);
    let mut v = Vec::new();
    assert_eq!(c.read_to_end(&mut v).unwrap(), 1);
    assert_eq!(v, b"1");

    let cap = if cfg!(miri) { 1024 } else { 1024 * 1024 };
    let data = (0..cap).map(|i| (i / 3) as u8).collect::<Vec<_>>();
    let mut v = Vec::new();
    let (a, b) = data.split_at(data.len() / 2);
    assert_eq!(Cursor::new(a).read_to_end(&mut v).unwrap(), a.len());
    assert_eq!(Cursor::new(b).read_to_end(&mut v).unwrap(), b.len());
    assert_eq!(v, data);
}

#[test]
fn read_to_string() {
    let mut c = Cursor::new(&b""[..]);
    let mut v = String::new();
    assert_eq!(c.read_to_string(&mut v).unwrap(), 0);
    assert_eq!(v, "");

    let mut c = Cursor::new(&b"1"[..]);
    let mut v = String::new();
    assert_eq!(c.read_to_string(&mut v).unwrap(), 1);
    assert_eq!(v, "1");

    let mut c = Cursor::new(&b"\xff"[..]);
    let mut v = String::new();
    assert!(c.read_to_string(&mut v).is_err());
}

#[test]
fn read_exact() {
    let mut buf = [0; 4];

    let mut c = Cursor::new(&b""[..]);
    assert_eq!(c.read_exact(&mut buf).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);

    let mut c = Cursor::new(&b"123"[..]).chain(Cursor::new(&b"456789"[..]));
    c.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"1234");
    c.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"5678");
    assert_eq!(c.read_exact(&mut buf).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
}

#[test]
fn read_exact_slice() {
    let mut buf = [0; 4];

    let mut c = &b""[..];
    assert_eq!(c.read_exact(&mut buf).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);

    let mut c = &b"123"[..];
    assert_eq!(c.read_exact(&mut buf).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    // make sure the optimized (early returning) method is being used
    assert_eq!(&buf, &[0; 4]);

    let mut c = &b"1234"[..];
    c.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"1234");

    let mut c = &b"56789"[..];
    c.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"5678");
    assert_eq!(c, b"9");
}

#[test]
fn read_buf_exact() {
    let buf: &mut [_] = &mut [0; 4];
    let mut buf: BorrowedBuf<'_> = buf.into();

    let mut c = Cursor::new(&b""[..]);
    assert_eq!(c.read_buf_exact(buf.unfilled()).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);

    let mut c = Cursor::new(&b"123456789"[..]);
    c.read_buf_exact(buf.unfilled()).unwrap();
    assert_eq!(buf.filled(), b"1234");

    buf.clear();

    c.read_buf_exact(buf.unfilled()).unwrap();
    assert_eq!(buf.filled(), b"5678");

    buf.clear();

    assert_eq!(c.read_buf_exact(buf.unfilled()).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
}

#[test]
#[should_panic]
fn borrowed_cursor_advance_overflow() {
    let mut buf = [0; 512];
    let mut buf = BorrowedBuf::from(&mut buf[..]);
    buf.unfilled().advance(1);
    buf.unfilled().advance(usize::MAX);
}

#[test]
fn take_eof() {
    struct R;

    impl Read for R {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::const_error!(io::ErrorKind::Other, ""))
        }
    }
    impl BufRead for R {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            Err(io::const_error!(io::ErrorKind::Other, ""))
        }
        fn consume(&mut self, _amt: usize) {}
    }

    let mut buf = [0; 1];
    assert_eq!(0, R.take(0).read(&mut buf).unwrap());
    assert_eq!(b"", R.take(0).fill_buf().unwrap());
}

fn cmp_bufread<Br1: BufRead, Br2: BufRead>(mut br1: Br1, mut br2: Br2, exp: &[u8]) {
    let mut cat = Vec::new();
    loop {
        let consume = {
            let buf1 = br1.fill_buf().unwrap();
            let buf2 = br2.fill_buf().unwrap();
            let minlen = if buf1.len() < buf2.len() { buf1.len() } else { buf2.len() };
            assert_eq!(buf1[..minlen], buf2[..minlen]);
            cat.extend_from_slice(&buf1[..minlen]);
            minlen
        };
        if consume == 0 {
            break;
        }
        br1.consume(consume);
        br2.consume(consume);
    }
    assert_eq!(br1.fill_buf().unwrap().len(), 0);
    assert_eq!(br2.fill_buf().unwrap().len(), 0);
    assert_eq!(&cat[..], &exp[..])
}

#[test]
fn chain_bufread() {
    let testdata = b"ABCDEFGHIJKL";
    let chain1 =
        (&testdata[..3]).chain(&testdata[3..6]).chain(&testdata[6..9]).chain(&testdata[9..]);
    let chain2 = (&testdata[..4]).chain(&testdata[4..8]).chain(&testdata[8..]);
    cmp_bufread(chain1, chain2, &testdata[..]);
}

#[test]
fn chain_splitted_char() {
    let chain = b"\xc3".chain(b"\xa9".as_slice());
    assert_eq!(crate::io::read_to_string(chain).unwrap(), "é");

    let mut chain = b"\xc3".chain(b"\xa9\n".as_slice());
    let mut buf = String::new();
    assert_eq!(chain.read_line(&mut buf).unwrap(), 3);
    assert_eq!(buf, "é\n");
}

#[test]
fn bufreader_size_hint() {
    let testdata = b"ABCDEFGHIJKL";
    let mut buf_reader = BufReader::new(&testdata[..]);
    assert_eq!(buf_reader.buffer().len(), 0);

    let buffer_length = testdata.len();
    buf_reader.fill_buf().unwrap();

    // Check that size hint matches buffer contents
    let mut buffered_bytes = buf_reader.bytes();
    let (lower_bound, _upper_bound) = buffered_bytes.size_hint();
    assert_eq!(lower_bound, buffer_length);

    // Check that size hint matches buffer contents after advancing
    buffered_bytes.next().unwrap().unwrap();
    let (lower_bound, _upper_bound) = buffered_bytes.size_hint();
    assert_eq!(lower_bound, buffer_length - 1);
}

#[test]
fn empty_size_hint() {
    let size_hint = io::empty().bytes().size_hint();
    assert_eq!(size_hint, (0, Some(0)));
}

#[test]
fn slice_size_hint() {
    let size_hint = (&[1, 2, 3]).bytes().size_hint();
    assert_eq!(size_hint, (3, Some(3)));
}

#[test]
fn take_size_hint() {
    let size_hint = (&[1, 2, 3]).take(2).bytes().size_hint();
    assert_eq!(size_hint, (2, Some(2)));

    let size_hint = (&[1, 2, 3]).take(4).bytes().size_hint();
    assert_eq!(size_hint, (3, Some(3)));

    let size_hint = io::repeat(0).take(3).bytes().size_hint();
    assert_eq!(size_hint, (3, Some(3)));
}

#[test]
fn chain_empty_size_hint() {
    let chain = io::empty().chain(io::empty());
    let size_hint = chain.bytes().size_hint();
    assert_eq!(size_hint, (0, Some(0)));
}

#[test]
fn chain_size_hint() {
    let testdata = b"ABCDEFGHIJKL";
    let mut buf_reader_1 = BufReader::new(&testdata[..6]);
    let mut buf_reader_2 = BufReader::new(&testdata[6..]);

    buf_reader_1.fill_buf().unwrap();
    buf_reader_2.fill_buf().unwrap();

    let chain = buf_reader_1.chain(buf_reader_2);
    let size_hint = chain.bytes().size_hint();
    assert_eq!(size_hint, (testdata.len(), Some(testdata.len())));
}

#[test]
fn chain_zero_length_read_is_not_eof() {
    let a = b"A";
    let b = b"B";
    let mut s = String::new();
    let mut chain = (&a[..]).chain(&b[..]);
    chain.read(&mut []).unwrap();
    chain.read_to_string(&mut s).unwrap();
    assert_eq!("AB", s);
}

#[bench]
#[cfg_attr(target_os = "emscripten", ignore)]
#[cfg_attr(miri, ignore)] // Miri isn't fast...
fn bench_read_to_end(b: &mut test::Bencher) {
    b.iter(|| {
        let mut lr = repeat(1).take(10000000);
        let mut vec = Vec::with_capacity(1024);
        super::default_read_to_end(&mut lr, &mut vec, None)
    });
}

#[test]
fn seek_len() -> io::Result<()> {
    let mut c = Cursor::new(vec![0; 15]);
    assert_eq!(c.stream_len()?, 15);

    c.seek(SeekFrom::End(0))?;
    let old_pos = c.stream_position()?;
    assert_eq!(c.stream_len()?, 15);
    assert_eq!(c.stream_position()?, old_pos);

    c.seek(SeekFrom::Start(7))?;
    c.seek(SeekFrom::Current(2))?;
    let old_pos = c.stream_position()?;
    assert_eq!(c.stream_len()?, 15);
    assert_eq!(c.stream_position()?, old_pos);

    Ok(())
}

#[test]
fn seek_position() -> io::Result<()> {
    // All `asserts` are duplicated here to make sure the method does not
    // change anything about the seek state.
    let mut c = Cursor::new(vec![0; 15]);
    assert_eq!(c.stream_position()?, 0);
    assert_eq!(c.stream_position()?, 0);

    c.seek(SeekFrom::End(0))?;
    assert_eq!(c.stream_position()?, 15);
    assert_eq!(c.stream_position()?, 15);

    c.seek(SeekFrom::Start(7))?;
    c.seek(SeekFrom::Current(2))?;
    assert_eq!(c.stream_position()?, 9);
    assert_eq!(c.stream_position()?, 9);

    c.seek(SeekFrom::End(-3))?;
    c.seek(SeekFrom::Current(1))?;
    c.seek(SeekFrom::Current(-5))?;
    assert_eq!(c.stream_position()?, 8);
    assert_eq!(c.stream_position()?, 8);

    c.rewind()?;
    assert_eq!(c.stream_position()?, 0);
    assert_eq!(c.stream_position()?, 0);

    Ok(())
}

// A simple example reader which uses the default implementation of
// read_to_end.
struct ExampleSliceReader<'a> {
    slice: &'a [u8],
}

impl<'a> Read for ExampleSliceReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let len = cmp::min(self.slice.len(), buf.len());
        buf[..len].copy_from_slice(&self.slice[..len]);
        self.slice = &self.slice[len..];
        Ok(len)
    }
}

#[test]
fn test_read_to_end_capacity() -> io::Result<()> {
    let input = &b"foo"[..];

    // read_to_end() takes care not to over-allocate when a buffer is the
    // exact size needed.
    let mut vec1 = Vec::with_capacity(input.len());
    ExampleSliceReader { slice: input }.read_to_end(&mut vec1)?;
    assert_eq!(vec1.len(), input.len());
    assert_eq!(vec1.capacity(), input.len(), "did not allocate more");

    Ok(())
}

#[test]
fn io_slice_mut_advance_slices() {
    let mut buf1 = [1; 8];
    let mut buf2 = [2; 16];
    let mut buf3 = [3; 8];
    let mut bufs = &mut [
        IoSliceMut::new(&mut buf1),
        IoSliceMut::new(&mut buf2),
        IoSliceMut::new(&mut buf3),
    ][..];

    // Only in a single buffer..
    IoSliceMut::advance_slices(&mut bufs, 1);
    assert_eq!(bufs[0].deref(), [1; 7].as_ref());
    assert_eq!(bufs[1].deref(), [2; 16].as_ref());
    assert_eq!(bufs[2].deref(), [3; 8].as_ref());

    // Removing a buffer, leaving others as is.
    IoSliceMut::advance_slices(&mut bufs, 7);
    assert_eq!(bufs[0].deref(), [2; 16].as_ref());
    assert_eq!(bufs[1].deref(), [3; 8].as_ref());

    // Removing a buffer and removing from the next buffer.
    IoSliceMut::advance_slices(&mut bufs, 18);
    assert_eq!(bufs[0].deref(), [3; 6].as_ref());
}

#[test]
#[should_panic]
fn io_slice_mut_advance_slices_empty_slice() {
    let mut empty_bufs = &mut [][..];
    IoSliceMut::advance_slices(&mut empty_bufs, 1);
}

#[test]
#[should_panic]
fn io_slice_mut_advance_slices_beyond_total_length() {
    let mut buf1 = [1; 8];
    let mut bufs = &mut [IoSliceMut::new(&mut buf1)][..];

    IoSliceMut::advance_slices(&mut bufs, 9);
    assert!(bufs.is_empty());
}

#[test]
fn io_slice_advance_slices() {
    let buf1 = [1; 8];
    let buf2 = [2; 16];
    let buf3 = [3; 8];
    let mut bufs = &mut [IoSlice::new(&buf1), IoSlice::new(&buf2), IoSlice::new(&buf3)][..];

    // Only in a single buffer..
    IoSlice::advance_slices(&mut bufs, 1);
    assert_eq!(bufs[0].deref(), [1; 7].as_ref());
    assert_eq!(bufs[1].deref(), [2; 16].as_ref());
    assert_eq!(bufs[2].deref(), [3; 8].as_ref());

    // Removing a buffer, leaving others as is.
    IoSlice::advance_slices(&mut bufs, 7);
    assert_eq!(bufs[0].deref(), [2; 16].as_ref());
    assert_eq!(bufs[1].deref(), [3; 8].as_ref());

    // Removing a buffer and removing from the next buffer.
    IoSlice::advance_slices(&mut bufs, 18);
    assert_eq!(bufs[0].deref(), [3; 6].as_ref());
}

#[test]
#[should_panic]
fn io_slice_advance_slices_empty_slice() {
    let mut empty_bufs = &mut [][..];
    IoSlice::advance_slices(&mut empty_bufs, 1);
}

#[test]
#[should_panic]
fn io_slice_advance_slices_beyond_total_length() {
    let buf1 = [1; 8];
    let mut bufs = &mut [IoSlice::new(&buf1)][..];

    IoSlice::advance_slices(&mut bufs, 9);
    assert!(bufs.is_empty());
}

#[test]
fn io_slice_as_slice() {
    let buf = [1; 8];
    let slice = IoSlice::new(&buf).as_slice();
    assert_eq!(slice, buf);
}

#[test]
fn io_slice_into_slice() {
    let mut buf = [1; 8];
    let slice = IoSliceMut::new(&mut buf).into_slice();
    assert_eq!(slice, [1; 8]);
}

/// Creates a new writer that reads from at most `n_bufs` and reads
/// `per_call` bytes (in total) per call to write.
fn test_writer(n_bufs: usize, per_call: usize) -> TestWriter {
    TestWriter { n_bufs, per_call, written: Vec::new() }
}

struct TestWriter {
    n_bufs: usize,
    per_call: usize,
    written: Vec<u8>,
}

impl Write for TestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.write_vectored(&[IoSlice::new(buf)])
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let mut left = self.per_call;
        let mut written = 0;
        for buf in bufs.iter().take(self.n_bufs) {
            let n = min(left, buf.len());
            self.written.extend_from_slice(&buf[0..n]);
            left -= n;
            written += n;
        }
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn test_writer_read_from_one_buf() {
    let mut writer = test_writer(1, 2);

    assert_eq!(writer.write(&[]).unwrap(), 0);
    assert_eq!(writer.write_vectored(&[]).unwrap(), 0);

    // Read at most 2 bytes.
    assert_eq!(writer.write(&[1, 1, 1]).unwrap(), 2);
    let bufs = &[IoSlice::new(&[2, 2, 2])];
    assert_eq!(writer.write_vectored(bufs).unwrap(), 2);

    // Only read from first buf.
    let bufs = &[IoSlice::new(&[3]), IoSlice::new(&[4, 4])];
    assert_eq!(writer.write_vectored(bufs).unwrap(), 1);

    assert_eq!(writer.written, &[1, 1, 2, 2, 3]);
}

#[test]
fn test_writer_read_from_multiple_bufs() {
    let mut writer = test_writer(3, 3);

    // Read at most 3 bytes from two buffers.
    let bufs = &[IoSlice::new(&[1]), IoSlice::new(&[2, 2, 2])];
    assert_eq!(writer.write_vectored(bufs).unwrap(), 3);

    // Read at most 3 bytes from three buffers.
    let bufs = &[IoSlice::new(&[3]), IoSlice::new(&[4]), IoSlice::new(&[5, 5])];
    assert_eq!(writer.write_vectored(bufs).unwrap(), 3);

    assert_eq!(writer.written, &[1, 2, 2, 3, 4, 5]);
}

#[test]
fn test_write_all_vectored() {
    #[rustfmt::skip] // Becomes unreadable otherwise.
    let tests: Vec<(_, &'static [u8])> = vec![
        (vec![], &[]),
        (vec![IoSlice::new(&[]), IoSlice::new(&[])], &[]),
        (vec![IoSlice::new(&[1])], &[1]),
        (vec![IoSlice::new(&[1, 2])], &[1, 2]),
        (vec![IoSlice::new(&[1, 2, 3])], &[1, 2, 3]),
        (vec![IoSlice::new(&[1, 2, 3, 4])], &[1, 2, 3, 4]),
        (vec![IoSlice::new(&[1, 2, 3, 4, 5])], &[1, 2, 3, 4, 5]),
        (vec![IoSlice::new(&[1]), IoSlice::new(&[2])], &[1, 2]),
        (vec![IoSlice::new(&[1]), IoSlice::new(&[2, 2])], &[1, 2, 2]),
        (vec![IoSlice::new(&[1, 1]), IoSlice::new(&[2, 2])], &[1, 1, 2, 2]),
        (vec![IoSlice::new(&[1, 1]), IoSlice::new(&[2, 2, 2])], &[1, 1, 2, 2, 2]),
        (vec![IoSlice::new(&[1, 1]), IoSlice::new(&[2, 2, 2])], &[1, 1, 2, 2, 2]),
        (vec![IoSlice::new(&[1, 1, 1]), IoSlice::new(&[2, 2, 2])], &[1, 1, 1, 2, 2, 2]),
        (vec![IoSlice::new(&[1, 1, 1]), IoSlice::new(&[2, 2, 2, 2])], &[1, 1, 1, 2, 2, 2, 2]),
        (vec![IoSlice::new(&[1, 1, 1, 1]), IoSlice::new(&[2, 2, 2, 2])], &[1, 1, 1, 1, 2, 2, 2, 2]),
        (vec![IoSlice::new(&[1]), IoSlice::new(&[2]), IoSlice::new(&[3])], &[1, 2, 3]),
        (vec![IoSlice::new(&[1, 1]), IoSlice::new(&[2, 2]), IoSlice::new(&[3, 3])], &[1, 1, 2, 2, 3, 3]),
        (vec![IoSlice::new(&[1]), IoSlice::new(&[2, 2]), IoSlice::new(&[3, 3, 3])], &[1, 2, 2, 3, 3, 3]),
        (vec![IoSlice::new(&[1, 1, 1]), IoSlice::new(&[2, 2, 2]), IoSlice::new(&[3, 3, 3])], &[1, 1, 1, 2, 2, 2, 3, 3, 3]),
    ];

    let writer_configs = &[(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)];

    for (n_bufs, per_call) in writer_configs.iter().copied() {
        for (mut input, wanted) in tests.clone().into_iter() {
            let mut writer = test_writer(n_bufs, per_call);
            assert!(writer.write_all_vectored(&mut *input).is_ok());
            assert_eq!(&*writer.written, &*wanted);
        }
    }
}

// Issue 94981
#[test]
#[should_panic = "number of read bytes exceeds limit"]
fn test_take_wrong_length() {
    struct LieAboutSize(bool);

    impl Read for LieAboutSize {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            // Lie about the read size at first time of read.
            if core::mem::take(&mut self.0) { Ok(buf.len() + 1) } else { Ok(buf.len()) }
        }
    }

    let mut buffer = vec![0; 4];
    let mut reader = LieAboutSize(true).take(4);
    // Primed the `Limit` by lying about the read size.
    let _ = reader.read(&mut buffer[..]);
}

#[test]
fn slice_read_exact_eof() {
    let slice = &b"123456"[..];

    let mut r = slice;
    assert!(r.read_exact(&mut [0; 10]).is_err());
    assert!(r.is_empty());

    let mut r = slice;
    let buf = &mut [0; 10];
    let mut buf = BorrowedBuf::from(buf.as_mut_slice());
    assert!(r.read_buf_exact(buf.unfilled()).is_err());
    assert!(r.is_empty());
    assert_eq!(buf.filled(), b"123456");
}

#[test]
fn cursor_read_exact_eof() {
    let slice = Cursor::new(b"123456");

    let mut r = slice.clone();
    assert!(r.read_exact(&mut [0; 10]).is_err());
    assert!(Cursor::split(&r).1.is_empty());

    let mut r = slice;
    let buf = &mut [0; 10];
    let mut buf = BorrowedBuf::from(buf.as_mut_slice());
    assert!(r.read_buf_exact(buf.unfilled()).is_err());
    assert!(Cursor::split(&r).1.is_empty());
    assert_eq!(buf.filled(), b"123456");
}

#[bench]
fn bench_take_read(b: &mut test::Bencher) {
    b.iter(|| {
        let mut buf = [0; 64];

        [255; 128].take(64).read(&mut buf).unwrap();
    });
}

#[bench]
fn bench_take_read_buf(b: &mut test::Bencher) {
    b.iter(|| {
        let buf: &mut [_] = &mut [MaybeUninit::uninit(); 64];

        let mut buf: BorrowedBuf<'_> = buf.into();

        [255; 128].take(64).read_buf(buf.unfilled()).unwrap();
    });
}

// Issue #120603
#[test]
#[should_panic]
fn read_buf_broken_read() {
    struct MalformedRead;

    impl Read for MalformedRead {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            // broken length calculation
            Ok(buf.len() + 1)
        }
    }

    let _ = BufReader::new(MalformedRead).fill_buf();
}

#[test]
fn read_buf_full_read() {
    struct FullRead;

    impl Read for FullRead {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            Ok(buf.len())
        }
    }

    assert_eq!(BufReader::new(FullRead).fill_buf().unwrap().len(), DEFAULT_BUF_SIZE);
}

struct DataAndErrorReader(&'static [u8]);

impl Read for DataAndErrorReader {
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        panic!("We want tests to use `read_buf`")
    }

    fn read_buf(&mut self, buf: io::BorrowedCursor<'_>) -> io::Result<()> {
        self.0.read_buf(buf).unwrap();
        Err(io::Error::other("error"))
    }
}

#[test]
fn read_buf_data_and_error_take() {
    let mut buf = [0; 64];
    let mut buf = io::BorrowedBuf::from(buf.as_mut_slice());

    let mut r = DataAndErrorReader(&[4, 5, 6]).take(1);
    assert!(r.read_buf(buf.unfilled()).is_err());
    assert_eq!(buf.filled(), &[4]);

    assert!(r.read_buf(buf.unfilled()).is_ok());
    assert_eq!(buf.filled(), &[4]);
    assert_eq!(r.get_ref().0, &[5, 6]);
}

#[test]
fn read_buf_data_and_error_buf() {
    let mut r = BufReader::new(DataAndErrorReader(&[4, 5, 6]));

    assert!(r.fill_buf().is_err());
    assert_eq!(r.fill_buf().unwrap(), &[4, 5, 6]);
}

#[test]
fn read_buf_data_and_error_read_to_end() {
    let mut r = DataAndErrorReader(&[4, 5, 6]);

    let mut v = Vec::with_capacity(200);
    assert!(r.read_to_end(&mut v).is_err());

    assert_eq!(v, &[4, 5, 6]);
}

#[test]
fn read_to_end_error() {
    struct ErrorReader;

    impl Read for ErrorReader {
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("error"))
        }
    }

    let mut r = [4, 5, 6].chain(ErrorReader);

    let mut v = Vec::with_capacity(200);
    assert!(r.read_to_end(&mut v).is_err());

    assert_eq!(v, &[4, 5, 6]);
}

#[test]
// Miri does not support signalling OOM
#[cfg_attr(miri, ignore)]
// 64-bit only to be sure the allocator will fail fast on an impossible to satisfy size
#[cfg(target_pointer_width = "64")]
fn try_oom_error() {
    let mut v = Vec::<u8>::new();
    let reserve_err = v.try_reserve(isize::MAX as usize - 1).unwrap_err();
    let io_err = io::Error::from(reserve_err);
    assert_eq!(io::ErrorKind::OutOfMemory, io_err.kind());
}
