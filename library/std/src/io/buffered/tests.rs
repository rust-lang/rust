use crate::io::prelude::*;
use crate::io::{self, BufReader, BufWriter, ErrorKind, IoSlice, LineWriter, ReadBuf, SeekFrom};
use crate::mem::MaybeUninit;
use crate::panic;
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::thread;

/// A dummy reader intended at testing short-reads propagation.
pub struct ShortReader {
    lengths: Vec<usize>,
}

// FIXME: rustfmt and tidy disagree about the correct formatting of this
// function. This leads to issues for users with editors configured to
// rustfmt-on-save.
impl Read for ShortReader {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        if self.lengths.is_empty() { Ok(0) } else { Ok(self.lengths.remove(0)) }
    }
}

#[test]
fn test_buffered_reader() {
    let inner: &[u8] = &[5, 6, 7, 0, 1, 2, 3, 4];
    let mut reader = BufReader::with_capacity(2, inner);

    let mut buf = [0, 0, 0];
    let nread = reader.read(&mut buf);
    assert_eq!(nread.unwrap(), 3);
    assert_eq!(buf, [5, 6, 7]);
    assert_eq!(reader.buffer(), []);

    let mut buf = [0, 0];
    let nread = reader.read(&mut buf);
    assert_eq!(nread.unwrap(), 2);
    assert_eq!(buf, [0, 1]);
    assert_eq!(reader.buffer(), []);

    let mut buf = [0];
    let nread = reader.read(&mut buf);
    assert_eq!(nread.unwrap(), 1);
    assert_eq!(buf, [2]);
    assert_eq!(reader.buffer(), [3]);

    let mut buf = [0, 0, 0];
    let nread = reader.read(&mut buf);
    assert_eq!(nread.unwrap(), 1);
    assert_eq!(buf, [3, 0, 0]);
    assert_eq!(reader.buffer(), []);

    let nread = reader.read(&mut buf);
    assert_eq!(nread.unwrap(), 1);
    assert_eq!(buf, [4, 0, 0]);
    assert_eq!(reader.buffer(), []);

    assert_eq!(reader.read(&mut buf).unwrap(), 0);
}

#[test]
fn test_buffered_reader_read_buf() {
    let inner: &[u8] = &[5, 6, 7, 0, 1, 2, 3, 4];
    let mut reader = BufReader::with_capacity(2, inner);

    let mut buf = [MaybeUninit::uninit(); 3];
    let mut buf = ReadBuf::uninit(&mut buf);

    reader.read_buf(&mut buf).unwrap();

    assert_eq!(buf.filled(), [5, 6, 7]);
    assert_eq!(reader.buffer(), []);

    let mut buf = [MaybeUninit::uninit(); 2];
    let mut buf = ReadBuf::uninit(&mut buf);

    reader.read_buf(&mut buf).unwrap();

    assert_eq!(buf.filled(), [0, 1]);
    assert_eq!(reader.buffer(), []);

    let mut buf = [MaybeUninit::uninit(); 1];
    let mut buf = ReadBuf::uninit(&mut buf);

    reader.read_buf(&mut buf).unwrap();

    assert_eq!(buf.filled(), [2]);
    assert_eq!(reader.buffer(), [3]);

    let mut buf = [MaybeUninit::uninit(); 3];
    let mut buf = ReadBuf::uninit(&mut buf);

    reader.read_buf(&mut buf).unwrap();

    assert_eq!(buf.filled(), [3]);
    assert_eq!(reader.buffer(), []);

    reader.read_buf(&mut buf).unwrap();

    assert_eq!(buf.filled(), [3, 4]);
    assert_eq!(reader.buffer(), []);

    buf.clear();

    reader.read_buf(&mut buf).unwrap();

    assert_eq!(buf.filled_len(), 0);
}

#[test]
fn test_buffered_reader_seek() {
    let inner: &[u8] = &[5, 6, 7, 0, 1, 2, 3, 4];
    let mut reader = BufReader::with_capacity(2, io::Cursor::new(inner));

    assert_eq!(reader.seek(SeekFrom::Start(3)).ok(), Some(3));
    assert_eq!(reader.fill_buf().ok(), Some(&[0, 1][..]));
    assert_eq!(reader.seek(SeekFrom::Current(0)).ok(), Some(3));
    assert_eq!(reader.fill_buf().ok(), Some(&[0, 1][..]));
    assert_eq!(reader.seek(SeekFrom::Current(1)).ok(), Some(4));
    assert_eq!(reader.fill_buf().ok(), Some(&[1, 2][..]));
    reader.consume(1);
    assert_eq!(reader.seek(SeekFrom::Current(-2)).ok(), Some(3));
}

#[test]
fn test_buffered_reader_seek_relative() {
    let inner: &[u8] = &[5, 6, 7, 0, 1, 2, 3, 4];
    let mut reader = BufReader::with_capacity(2, io::Cursor::new(inner));

    assert!(reader.seek_relative(3).is_ok());
    assert_eq!(reader.fill_buf().ok(), Some(&[0, 1][..]));
    assert!(reader.seek_relative(0).is_ok());
    assert_eq!(reader.fill_buf().ok(), Some(&[0, 1][..]));
    assert!(reader.seek_relative(1).is_ok());
    assert_eq!(reader.fill_buf().ok(), Some(&[1][..]));
    assert!(reader.seek_relative(-1).is_ok());
    assert_eq!(reader.fill_buf().ok(), Some(&[0, 1][..]));
    assert!(reader.seek_relative(2).is_ok());
    assert_eq!(reader.fill_buf().ok(), Some(&[2, 3][..]));
}

#[test]
fn test_buffered_reader_stream_position() {
    let inner: &[u8] = &[5, 6, 7, 0, 1, 2, 3, 4];
    let mut reader = BufReader::with_capacity(2, io::Cursor::new(inner));

    assert_eq!(reader.stream_position().ok(), Some(0));
    assert_eq!(reader.seek(SeekFrom::Start(3)).ok(), Some(3));
    assert_eq!(reader.stream_position().ok(), Some(3));
    // relative seeking within the buffer and reading position should keep the buffer
    assert_eq!(reader.fill_buf().ok(), Some(&[0, 1][..]));
    assert!(reader.seek_relative(0).is_ok());
    assert_eq!(reader.stream_position().ok(), Some(3));
    assert_eq!(reader.buffer(), &[0, 1][..]);
    assert!(reader.seek_relative(1).is_ok());
    assert_eq!(reader.stream_position().ok(), Some(4));
    assert_eq!(reader.buffer(), &[1][..]);
    assert!(reader.seek_relative(-1).is_ok());
    assert_eq!(reader.stream_position().ok(), Some(3));
    assert_eq!(reader.buffer(), &[0, 1][..]);
    // relative seeking outside the buffer will discard it
    assert!(reader.seek_relative(2).is_ok());
    assert_eq!(reader.stream_position().ok(), Some(5));
    assert_eq!(reader.buffer(), &[][..]);
}

#[test]
fn test_buffered_reader_stream_position_panic() {
    let inner: &[u8] = &[5, 6, 7, 0, 1, 2, 3, 4];
    let mut reader = BufReader::with_capacity(4, io::Cursor::new(inner));

    // cause internal buffer to be filled but read only partially
    let mut buffer = [0, 0];
    assert!(reader.read_exact(&mut buffer).is_ok());
    // rewinding the internal reader will cause buffer to loose sync
    let inner = reader.get_mut();
    assert!(inner.seek(SeekFrom::Start(0)).is_ok());
    // overflow when subtracting the remaining buffer size from current position
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| reader.stream_position().ok()));
    assert!(result.is_err());
}

#[test]
fn test_buffered_reader_invalidated_after_read() {
    let inner: &[u8] = &[5, 6, 7, 0, 1, 2, 3, 4];
    let mut reader = BufReader::with_capacity(3, io::Cursor::new(inner));

    assert_eq!(reader.fill_buf().ok(), Some(&[5, 6, 7][..]));
    reader.consume(3);

    let mut buffer = [0, 0, 0, 0, 0];
    assert_eq!(reader.read(&mut buffer).ok(), Some(5));
    assert_eq!(buffer, [0, 1, 2, 3, 4]);

    assert!(reader.seek_relative(-2).is_ok());
    let mut buffer = [0, 0];
    assert_eq!(reader.read(&mut buffer).ok(), Some(2));
    assert_eq!(buffer, [3, 4]);
}

#[test]
fn test_buffered_reader_invalidated_after_seek() {
    let inner: &[u8] = &[5, 6, 7, 0, 1, 2, 3, 4];
    let mut reader = BufReader::with_capacity(3, io::Cursor::new(inner));

    assert_eq!(reader.fill_buf().ok(), Some(&[5, 6, 7][..]));
    reader.consume(3);

    assert!(reader.seek(SeekFrom::Current(5)).is_ok());

    assert!(reader.seek_relative(-2).is_ok());
    let mut buffer = [0, 0];
    assert_eq!(reader.read(&mut buffer).ok(), Some(2));
    assert_eq!(buffer, [3, 4]);
}

#[test]
fn test_buffered_reader_seek_underflow() {
    // gimmick reader that yields its position modulo 256 for each byte
    struct PositionReader {
        pos: u64,
    }
    impl Read for PositionReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let len = buf.len();
            for x in buf {
                *x = self.pos as u8;
                self.pos = self.pos.wrapping_add(1);
            }
            Ok(len)
        }
    }
    impl Seek for PositionReader {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            match pos {
                SeekFrom::Start(n) => {
                    self.pos = n;
                }
                SeekFrom::Current(n) => {
                    self.pos = self.pos.wrapping_add(n as u64);
                }
                SeekFrom::End(n) => {
                    self.pos = u64::MAX.wrapping_add(n as u64);
                }
            }
            Ok(self.pos)
        }
    }

    let mut reader = BufReader::with_capacity(5, PositionReader { pos: 0 });
    assert_eq!(reader.fill_buf().ok(), Some(&[0, 1, 2, 3, 4][..]));
    assert_eq!(reader.seek(SeekFrom::End(-5)).ok(), Some(u64::MAX - 5));
    assert_eq!(reader.fill_buf().ok().map(|s| s.len()), Some(5));
    // the following seek will require two underlying seeks
    let expected = 9223372036854775802;
    assert_eq!(reader.seek(SeekFrom::Current(i64::MIN)).ok(), Some(expected));
    assert_eq!(reader.fill_buf().ok().map(|s| s.len()), Some(5));
    // seeking to 0 should empty the buffer.
    assert_eq!(reader.seek(SeekFrom::Current(0)).ok(), Some(expected));
    assert_eq!(reader.get_ref().pos, expected);
}

#[test]
fn test_buffered_reader_seek_underflow_discard_buffer_between_seeks() {
    // gimmick reader that returns Err after first seek
    struct ErrAfterFirstSeekReader {
        first_seek: bool,
    }
    impl Read for ErrAfterFirstSeekReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            for x in &mut *buf {
                *x = 0;
            }
            Ok(buf.len())
        }
    }
    impl Seek for ErrAfterFirstSeekReader {
        fn seek(&mut self, _: SeekFrom) -> io::Result<u64> {
            if self.first_seek {
                self.first_seek = false;
                Ok(0)
            } else {
                Err(io::Error::new(io::ErrorKind::Other, "oh no!"))
            }
        }
    }

    let mut reader = BufReader::with_capacity(5, ErrAfterFirstSeekReader { first_seek: true });
    assert_eq!(reader.fill_buf().ok(), Some(&[0, 0, 0, 0, 0][..]));

    // The following seek will require two underlying seeks.  The first will
    // succeed but the second will fail.  This should still invalidate the
    // buffer.
    assert!(reader.seek(SeekFrom::Current(i64::MIN)).is_err());
    assert_eq!(reader.buffer().len(), 0);
}

#[test]
fn test_buffered_reader_read_to_end_consumes_buffer() {
    let data: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7];
    let mut reader = BufReader::with_capacity(3, data);
    let mut buf = Vec::new();
    assert_eq!(reader.fill_buf().ok(), Some(&[0, 1, 2][..]));
    assert_eq!(reader.read_to_end(&mut buf).ok(), Some(8));
    assert_eq!(&buf, &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert!(reader.buffer().is_empty());
}

#[test]
fn test_buffered_reader_read_to_string_consumes_buffer() {
    let data: &[u8] = "deadbeef".as_bytes();
    let mut reader = BufReader::with_capacity(3, data);
    let mut buf = String::new();
    assert_eq!(reader.fill_buf().ok(), Some("dea".as_bytes()));
    assert_eq!(reader.read_to_string(&mut buf).ok(), Some(8));
    assert_eq!(&buf, "deadbeef");
    assert!(reader.buffer().is_empty());
}

#[test]
fn test_buffered_writer() {
    let inner = Vec::new();
    let mut writer = BufWriter::with_capacity(2, inner);

    writer.write(&[0, 1]).unwrap();
    assert_eq!(writer.buffer(), []);
    assert_eq!(*writer.get_ref(), [0, 1]);

    writer.write(&[2]).unwrap();
    assert_eq!(writer.buffer(), [2]);
    assert_eq!(*writer.get_ref(), [0, 1]);

    writer.write(&[3]).unwrap();
    assert_eq!(writer.buffer(), [2, 3]);
    assert_eq!(*writer.get_ref(), [0, 1]);

    writer.flush().unwrap();
    assert_eq!(writer.buffer(), []);
    assert_eq!(*writer.get_ref(), [0, 1, 2, 3]);

    writer.write(&[4]).unwrap();
    writer.write(&[5]).unwrap();
    assert_eq!(writer.buffer(), [4, 5]);
    assert_eq!(*writer.get_ref(), [0, 1, 2, 3]);

    writer.write(&[6]).unwrap();
    assert_eq!(writer.buffer(), [6]);
    assert_eq!(*writer.get_ref(), [0, 1, 2, 3, 4, 5]);

    writer.write(&[7, 8]).unwrap();
    assert_eq!(writer.buffer(), []);
    assert_eq!(*writer.get_ref(), [0, 1, 2, 3, 4, 5, 6, 7, 8]);

    writer.write(&[9, 10, 11]).unwrap();
    assert_eq!(writer.buffer(), []);
    assert_eq!(*writer.get_ref(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);

    writer.flush().unwrap();
    assert_eq!(writer.buffer(), []);
    assert_eq!(*writer.get_ref(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
}

#[test]
fn test_buffered_writer_inner_flushes() {
    let mut w = BufWriter::with_capacity(3, Vec::new());
    w.write(&[0, 1]).unwrap();
    assert_eq!(*w.get_ref(), []);
    let w = w.into_inner().unwrap();
    assert_eq!(w, [0, 1]);
}

#[test]
fn test_buffered_writer_seek() {
    let mut w = BufWriter::with_capacity(3, io::Cursor::new(Vec::new()));
    w.write_all(&[0, 1, 2, 3, 4, 5]).unwrap();
    w.write_all(&[6, 7]).unwrap();
    assert_eq!(w.seek(SeekFrom::Current(0)).ok(), Some(8));
    assert_eq!(&w.get_ref().get_ref()[..], &[0, 1, 2, 3, 4, 5, 6, 7][..]);
    assert_eq!(w.seek(SeekFrom::Start(2)).ok(), Some(2));
    w.write_all(&[8, 9]).unwrap();
    assert_eq!(&w.into_inner().unwrap().into_inner()[..], &[0, 1, 8, 9, 4, 5, 6, 7]);
}

#[test]
fn test_read_until() {
    let inner: &[u8] = &[0, 1, 2, 1, 0];
    let mut reader = BufReader::with_capacity(2, inner);
    let mut v = Vec::new();
    reader.read_until(0, &mut v).unwrap();
    assert_eq!(v, [0]);
    v.truncate(0);
    reader.read_until(2, &mut v).unwrap();
    assert_eq!(v, [1, 2]);
    v.truncate(0);
    reader.read_until(1, &mut v).unwrap();
    assert_eq!(v, [1]);
    v.truncate(0);
    reader.read_until(8, &mut v).unwrap();
    assert_eq!(v, [0]);
    v.truncate(0);
    reader.read_until(9, &mut v).unwrap();
    assert_eq!(v, []);
}

#[test]
fn test_line_buffer() {
    let mut writer = LineWriter::new(Vec::new());
    writer.write(&[0]).unwrap();
    assert_eq!(*writer.get_ref(), []);
    writer.write(&[1]).unwrap();
    assert_eq!(*writer.get_ref(), []);
    writer.flush().unwrap();
    assert_eq!(*writer.get_ref(), [0, 1]);
    writer.write(&[0, b'\n', 1, b'\n', 2]).unwrap();
    assert_eq!(*writer.get_ref(), [0, 1, 0, b'\n', 1, b'\n']);
    writer.flush().unwrap();
    assert_eq!(*writer.get_ref(), [0, 1, 0, b'\n', 1, b'\n', 2]);
    writer.write(&[3, b'\n']).unwrap();
    assert_eq!(*writer.get_ref(), [0, 1, 0, b'\n', 1, b'\n', 2, 3, b'\n']);
}

#[test]
fn test_read_line() {
    let in_buf: &[u8] = b"a\nb\nc";
    let mut reader = BufReader::with_capacity(2, in_buf);
    let mut s = String::new();
    reader.read_line(&mut s).unwrap();
    assert_eq!(s, "a\n");
    s.truncate(0);
    reader.read_line(&mut s).unwrap();
    assert_eq!(s, "b\n");
    s.truncate(0);
    reader.read_line(&mut s).unwrap();
    assert_eq!(s, "c");
    s.truncate(0);
    reader.read_line(&mut s).unwrap();
    assert_eq!(s, "");
}

#[test]
fn test_lines() {
    let in_buf: &[u8] = b"a\nb\nc";
    let reader = BufReader::with_capacity(2, in_buf);
    let mut it = reader.lines();
    assert_eq!(it.next().unwrap().unwrap(), "a".to_string());
    assert_eq!(it.next().unwrap().unwrap(), "b".to_string());
    assert_eq!(it.next().unwrap().unwrap(), "c".to_string());
    assert!(it.next().is_none());
}

#[test]
fn test_short_reads() {
    let inner = ShortReader { lengths: vec![0, 1, 2, 0, 1, 0] };
    let mut reader = BufReader::new(inner);
    let mut buf = [0, 0];
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
    assert_eq!(reader.read(&mut buf).unwrap(), 1);
    assert_eq!(reader.read(&mut buf).unwrap(), 2);
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
    assert_eq!(reader.read(&mut buf).unwrap(), 1);
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
}

#[test]
#[should_panic]
fn dont_panic_in_drop_on_panicked_flush() {
    struct FailFlushWriter;

    impl Write for FailFlushWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::last_os_error())
        }
    }

    let writer = FailFlushWriter;
    let _writer = BufWriter::new(writer);

    // If writer panics *again* due to the flush error then the process will
    // abort.
    panic!();
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn panic_in_write_doesnt_flush_in_drop() {
    static WRITES: AtomicUsize = AtomicUsize::new(0);

    struct PanicWriter;

    impl Write for PanicWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            WRITES.fetch_add(1, Ordering::SeqCst);
            panic!();
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    thread::spawn(|| {
        let mut writer = BufWriter::new(PanicWriter);
        let _ = writer.write(b"hello world");
        let _ = writer.flush();
    })
    .join()
    .unwrap_err();

    assert_eq!(WRITES.load(Ordering::SeqCst), 1);
}

#[bench]
fn bench_buffered_reader(b: &mut test::Bencher) {
    b.iter(|| BufReader::new(io::empty()));
}

#[bench]
fn bench_buffered_reader_small_reads(b: &mut test::Bencher) {
    let data = (0..u8::MAX).cycle().take(1024 * 4).collect::<Vec<_>>();
    b.iter(|| {
        let mut reader = BufReader::new(&data[..]);
        let mut buf = [0u8; 4];
        for _ in 0..1024 {
            reader.read_exact(&mut buf).unwrap();
        }
    });
}

#[bench]
fn bench_buffered_writer(b: &mut test::Bencher) {
    b.iter(|| BufWriter::new(io::sink()));
}

/// A simple `Write` target, designed to be wrapped by `LineWriter` /
/// `BufWriter` / etc, that can have its `write` & `flush` behavior
/// configured
#[derive(Default, Clone)]
struct ProgrammableSink {
    // Writes append to this slice
    pub buffer: Vec<u8>,

    // If true, writes will always be an error
    pub always_write_error: bool,

    // If true, flushes will always be an error
    pub always_flush_error: bool,

    // If set, only up to this number of bytes will be written in a single
    // call to `write`
    pub accept_prefix: Option<usize>,

    // If set, counts down with each write, and writes return an error
    // when it hits 0
    pub max_writes: Option<usize>,

    // If set, attempting to write when max_writes == Some(0) will be an
    // error; otherwise, it will return Ok(0).
    pub error_after_max_writes: bool,
}

impl Write for ProgrammableSink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if self.always_write_error {
            return Err(io::Error::new(io::ErrorKind::Other, "test - always_write_error"));
        }

        match self.max_writes {
            Some(0) if self.error_after_max_writes => {
                return Err(io::Error::new(io::ErrorKind::Other, "test - max_writes"));
            }
            Some(0) => return Ok(0),
            Some(ref mut count) => *count -= 1,
            None => {}
        }

        let len = match self.accept_prefix {
            None => data.len(),
            Some(prefix) => data.len().min(prefix),
        };

        let data = &data[..len];
        self.buffer.extend_from_slice(data);

        Ok(len)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.always_flush_error {
            Err(io::Error::new(io::ErrorKind::Other, "test - always_flush_error"))
        } else {
            Ok(())
        }
    }
}

/// Previously the `LineWriter` could successfully write some bytes but
/// then fail to report that it has done so. Additionally, an erroneous
/// flush after a successful write was permanently ignored.
///
/// Test that a line writer correctly reports the number of written bytes,
/// and that it attempts to flush buffered lines from previous writes
/// before processing new data
///
/// Regression test for #37807
#[test]
fn erroneous_flush_retried() {
    let writer = ProgrammableSink {
        // Only write up to 4 bytes at a time
        accept_prefix: Some(4),

        // Accept the first two writes, then error the others
        max_writes: Some(2),
        error_after_max_writes: true,

        ..Default::default()
    };

    // This should write the first 4 bytes. The rest will be buffered, out
    // to the last newline.
    let mut writer = LineWriter::new(writer);
    assert_eq!(writer.write(b"a\nb\nc\nd\ne").unwrap(), 8);

    // This write should attempt to flush "c\nd\n", then buffer "e". No
    // errors should happen here because no further writes should be
    // attempted against `writer`.
    assert_eq!(writer.write(b"e").unwrap(), 1);
    assert_eq!(&writer.get_ref().buffer, b"a\nb\nc\nd\n");
}

#[test]
fn line_vectored() {
    let mut a = LineWriter::new(Vec::new());
    assert_eq!(
        a.write_vectored(&[
            IoSlice::new(&[]),
            IoSlice::new(b"\n"),
            IoSlice::new(&[]),
            IoSlice::new(b"a"),
        ])
        .unwrap(),
        2,
    );
    assert_eq!(a.get_ref(), b"\n");

    assert_eq!(
        a.write_vectored(&[
            IoSlice::new(&[]),
            IoSlice::new(b"b"),
            IoSlice::new(&[]),
            IoSlice::new(b"a"),
            IoSlice::new(&[]),
            IoSlice::new(b"c"),
        ])
        .unwrap(),
        3,
    );
    assert_eq!(a.get_ref(), b"\n");
    a.flush().unwrap();
    assert_eq!(a.get_ref(), b"\nabac");
    assert_eq!(a.write_vectored(&[]).unwrap(), 0);
    assert_eq!(
        a.write_vectored(&[
            IoSlice::new(&[]),
            IoSlice::new(&[]),
            IoSlice::new(&[]),
            IoSlice::new(&[]),
        ])
        .unwrap(),
        0,
    );
    assert_eq!(a.write_vectored(&[IoSlice::new(b"a\nb"),]).unwrap(), 3);
    assert_eq!(a.get_ref(), b"\nabaca\nb");
}

#[test]
fn line_vectored_partial_and_errors() {
    use crate::collections::VecDeque;

    enum Call {
        Write { inputs: Vec<&'static [u8]>, output: io::Result<usize> },
        Flush { output: io::Result<()> },
    }

    #[derive(Default)]
    struct Writer {
        calls: VecDeque<Call>,
    }

    impl Write for Writer {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.write_vectored(&[IoSlice::new(buf)])
        }

        fn write_vectored(&mut self, buf: &[IoSlice<'_>]) -> io::Result<usize> {
            match self.calls.pop_front().expect("unexpected call to write") {
                Call::Write { inputs, output } => {
                    assert_eq!(inputs, buf.iter().map(|b| &**b).collect::<Vec<_>>());
                    output
                }
                Call::Flush { .. } => panic!("unexpected call to write; expected a flush"),
            }
        }

        fn is_write_vectored(&self) -> bool {
            true
        }

        fn flush(&mut self) -> io::Result<()> {
            match self.calls.pop_front().expect("Unexpected call to flush") {
                Call::Flush { output } => output,
                Call::Write { .. } => panic!("unexpected call to flush; expected a write"),
            }
        }
    }

    impl Drop for Writer {
        fn drop(&mut self) {
            if !thread::panicking() {
                assert_eq!(self.calls.len(), 0);
            }
        }
    }

    // partial writes keep going
    let mut a = LineWriter::new(Writer::default());
    a.write_vectored(&[IoSlice::new(&[]), IoSlice::new(b"abc")]).unwrap();

    a.get_mut().calls.push_back(Call::Write { inputs: vec![b"abc"], output: Ok(1) });
    a.get_mut().calls.push_back(Call::Write { inputs: vec![b"bc"], output: Ok(2) });
    a.get_mut().calls.push_back(Call::Write { inputs: vec![b"x", b"\n"], output: Ok(2) });

    a.write_vectored(&[IoSlice::new(b"x"), IoSlice::new(b"\n")]).unwrap();

    a.get_mut().calls.push_back(Call::Flush { output: Ok(()) });
    a.flush().unwrap();

    // erroneous writes stop and don't write more
    a.get_mut().calls.push_back(Call::Write { inputs: vec![b"x", b"\na"], output: Err(err()) });
    a.get_mut().calls.push_back(Call::Flush { output: Ok(()) });
    assert!(a.write_vectored(&[IoSlice::new(b"x"), IoSlice::new(b"\na")]).is_err());
    a.flush().unwrap();

    fn err() -> io::Error {
        io::Error::new(io::ErrorKind::Other, "x")
    }
}

/// Test that, in cases where vectored writing is not enabled, the
/// LineWriter uses the normal `write` call, which more-correctly handles
/// partial lines
#[test]
fn line_vectored_ignored() {
    let writer = ProgrammableSink::default();
    let mut writer = LineWriter::new(writer);

    let content = [
        IoSlice::new(&[]),
        IoSlice::new(b"Line 1\nLine"),
        IoSlice::new(b" 2\nLine 3\nL"),
        IoSlice::new(&[]),
        IoSlice::new(&[]),
        IoSlice::new(b"ine 4"),
        IoSlice::new(b"\nLine 5\n"),
    ];

    let count = writer.write_vectored(&content).unwrap();
    assert_eq!(count, 11);
    assert_eq!(&writer.get_ref().buffer, b"Line 1\n");

    let count = writer.write_vectored(&content[2..]).unwrap();
    assert_eq!(count, 11);
    assert_eq!(&writer.get_ref().buffer, b"Line 1\nLine 2\nLine 3\n");

    let count = writer.write_vectored(&content[5..]).unwrap();
    assert_eq!(count, 5);
    assert_eq!(&writer.get_ref().buffer, b"Line 1\nLine 2\nLine 3\n");

    let count = writer.write_vectored(&content[6..]).unwrap();
    assert_eq!(count, 8);
    assert_eq!(
        writer.get_ref().buffer.as_slice(),
        b"Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n".as_ref()
    );
}

/// Test that, given this input:
///
/// Line 1\n
/// Line 2\n
/// Line 3\n
/// Line 4
///
/// And given a result that only writes to midway through Line 2
///
/// That only up to the end of Line 3 is buffered
///
/// This behavior is desirable because it prevents flushing partial lines
#[test]
fn partial_write_buffers_line() {
    let writer = ProgrammableSink { accept_prefix: Some(13), ..Default::default() };
    let mut writer = LineWriter::new(writer);

    assert_eq!(writer.write(b"Line 1\nLine 2\nLine 3\nLine4").unwrap(), 21);
    assert_eq!(&writer.get_ref().buffer, b"Line 1\nLine 2");

    assert_eq!(writer.write(b"Line 4").unwrap(), 6);
    assert_eq!(&writer.get_ref().buffer, b"Line 1\nLine 2\nLine 3\n");
}

/// Test that, given this input:
///
/// Line 1\n
/// Line 2\n
/// Line 3
///
/// And given that the full write of lines 1 and 2 was successful
/// That data up to Line 3 is buffered
#[test]
fn partial_line_buffered_after_line_write() {
    let writer = ProgrammableSink::default();
    let mut writer = LineWriter::new(writer);

    assert_eq!(writer.write(b"Line 1\nLine 2\nLine 3").unwrap(), 20);
    assert_eq!(&writer.get_ref().buffer, b"Line 1\nLine 2\n");

    assert!(writer.flush().is_ok());
    assert_eq!(&writer.get_ref().buffer, b"Line 1\nLine 2\nLine 3");
}

/// Test that, given a partial line that exceeds the length of
/// LineBuffer's buffer (that is, without a trailing newline), that that
/// line is written to the inner writer
#[test]
fn long_line_flushed() {
    let writer = ProgrammableSink::default();
    let mut writer = LineWriter::with_capacity(5, writer);

    assert_eq!(writer.write(b"0123456789").unwrap(), 10);
    assert_eq!(&writer.get_ref().buffer, b"0123456789");
}

/// Test that, given a very long partial line *after* successfully
/// flushing a complete line, that that line is buffered unconditionally,
/// and no additional writes take place. This assures the property that
/// `write` should make at-most-one attempt to write new data.
#[test]
fn line_long_tail_not_flushed() {
    let writer = ProgrammableSink::default();
    let mut writer = LineWriter::with_capacity(5, writer);

    // Assert that Line 1\n is flushed, and 01234 is buffered
    assert_eq!(writer.write(b"Line 1\n0123456789").unwrap(), 12);
    assert_eq!(&writer.get_ref().buffer, b"Line 1\n");

    // Because the buffer is full, this subsequent write will flush it
    assert_eq!(writer.write(b"5").unwrap(), 1);
    assert_eq!(&writer.get_ref().buffer, b"Line 1\n01234");
}

/// Test that, if an attempt to pre-flush buffered data returns Ok(0),
/// this is propagated as an error.
#[test]
fn line_buffer_write0_error() {
    let writer = ProgrammableSink {
        // Accept one write, then return Ok(0) on subsequent ones
        max_writes: Some(1),

        ..Default::default()
    };
    let mut writer = LineWriter::new(writer);

    // This should write "Line 1\n" and buffer "Partial"
    assert_eq!(writer.write(b"Line 1\nPartial").unwrap(), 14);
    assert_eq!(&writer.get_ref().buffer, b"Line 1\n");

    // This will attempt to flush "partial", which will return Ok(0), which
    // needs to be an error, because we've already informed the client
    // that we accepted the write.
    let err = writer.write(b" Line End\n").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::WriteZero);
    assert_eq!(&writer.get_ref().buffer, b"Line 1\n");
}

/// Test that, if a write returns Ok(0) after a successful pre-flush, this
/// is propagated as Ok(0)
#[test]
fn line_buffer_write0_normal() {
    let writer = ProgrammableSink {
        // Accept two writes, then return Ok(0) on subsequent ones
        max_writes: Some(2),

        ..Default::default()
    };
    let mut writer = LineWriter::new(writer);

    // This should write "Line 1\n" and buffer "Partial"
    assert_eq!(writer.write(b"Line 1\nPartial").unwrap(), 14);
    assert_eq!(&writer.get_ref().buffer, b"Line 1\n");

    // This will flush partial, which will succeed, but then return Ok(0)
    // when flushing " Line End\n"
    assert_eq!(writer.write(b" Line End\n").unwrap(), 0);
    assert_eq!(&writer.get_ref().buffer, b"Line 1\nPartial");
}

/// LineWriter has a custom `write_all`; make sure it works correctly
#[test]
fn line_write_all() {
    let writer = ProgrammableSink {
        // Only write 5 bytes at a time
        accept_prefix: Some(5),
        ..Default::default()
    };
    let mut writer = LineWriter::new(writer);

    writer.write_all(b"Line 1\nLine 2\nLine 3\nLine 4\nPartial").unwrap();
    assert_eq!(&writer.get_ref().buffer, b"Line 1\nLine 2\nLine 3\nLine 4\n");
    writer.write_all(b" Line 5\n").unwrap();
    assert_eq!(
        writer.get_ref().buffer.as_slice(),
        b"Line 1\nLine 2\nLine 3\nLine 4\nPartial Line 5\n".as_ref(),
    );
}

#[test]
fn line_write_all_error() {
    let writer = ProgrammableSink {
        // Only accept up to 3 writes of up to 5 bytes each
        accept_prefix: Some(5),
        max_writes: Some(3),
        ..Default::default()
    };

    let mut writer = LineWriter::new(writer);
    let res = writer.write_all(b"Line 1\nLine 2\nLine 3\nLine 4\nPartial");
    assert!(res.is_err());
    // An error from write_all leaves everything in an indeterminate state,
    // so there's nothing else to test here
}

/// Under certain circumstances, the old implementation of LineWriter
/// would try to buffer "to the last newline" but be forced to buffer
/// less than that, leading to inappropriate partial line writes.
/// Regression test for that issue.
#[test]
fn partial_multiline_buffering() {
    let writer = ProgrammableSink {
        // Write only up to 5 bytes at a time
        accept_prefix: Some(5),
        ..Default::default()
    };

    let mut writer = LineWriter::with_capacity(10, writer);

    let content = b"AAAAABBBBB\nCCCCDDDDDD\nEEE";

    // When content is written, LineWriter will try to write blocks A, B,
    // C, and D. Only block A will succeed. Under the old behavior, LineWriter
    // would then try to buffer B, C and D, but because its capacity is 10,
    // it will only be able to buffer B and C. We don't want to buffer
    // partial lines concurrent with whole lines, so the correct behavior
    // is to buffer only block B (out to the newline)
    assert_eq!(writer.write(content).unwrap(), 11);
    assert_eq!(writer.get_ref().buffer, *b"AAAAA");

    writer.flush().unwrap();
    assert_eq!(writer.get_ref().buffer, *b"AAAAABBBBB\n");
}

/// Same as test_partial_multiline_buffering, but in the event NO full lines
/// fit in the buffer, just buffer as much as possible
#[test]
fn partial_multiline_buffering_without_full_line() {
    let writer = ProgrammableSink {
        // Write only up to 5 bytes at a time
        accept_prefix: Some(5),
        ..Default::default()
    };

    let mut writer = LineWriter::with_capacity(5, writer);

    let content = b"AAAAABBBBBBBBBB\nCCCCC\nDDDDD";

    // When content is written, LineWriter will try to write blocks A, B,
    // and C. Only block A will succeed. Under the old behavior, LineWriter
    // would then try to buffer B and C, but because its capacity is 5,
    // it will only be able to buffer part of B. Because it's not possible
    // for it to buffer any complete lines, it should buffer as much of B as
    // possible
    assert_eq!(writer.write(content).unwrap(), 10);
    assert_eq!(writer.get_ref().buffer, *b"AAAAA");

    writer.flush().unwrap();
    assert_eq!(writer.get_ref().buffer, *b"AAAAABBBBB");
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RecordedEvent {
    Write(String),
    Flush,
}

#[derive(Debug, Clone, Default)]
struct WriteRecorder {
    pub events: Vec<RecordedEvent>,
}

impl Write for WriteRecorder {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        use crate::str::from_utf8;

        self.events.push(RecordedEvent::Write(from_utf8(buf).unwrap().to_string()));
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.events.push(RecordedEvent::Flush);
        Ok(())
    }
}

/// Test that a normal, formatted writeln only results in a single write
/// call to the underlying writer. A naive implementation of
/// LineWriter::write_all results in two writes: one of the buffered data,
/// and another of the final substring in the formatted set
#[test]
fn single_formatted_write() {
    let writer = WriteRecorder::default();
    let mut writer = LineWriter::new(writer);

    // Under a naive implementation of LineWriter, this will result in two
    // writes: "hello, world" and "!\n", because write() has to flush the
    // buffer before attempting to write the last "!\n". write_all shouldn't
    // have this limitation.
    writeln!(&mut writer, "{}, {}!", "hello", "world").unwrap();
    assert_eq!(writer.get_ref().events, [RecordedEvent::Write("hello, world!\n".to_string())]);
}
