use crate::cmp::{max, min};
use crate::io::prelude::*;
use crate::io::{
    copy, empty, repeat, sink, BorrowedBuf, BufWriter, Empty, Repeat, Result, SeekFrom, Sink,
    DEFAULT_BUF_SIZE,
};

use crate::mem::MaybeUninit;

#[test]
fn copy_copies() {
    let mut r = repeat(0).take(4);
    let mut w = sink();
    assert_eq!(copy(&mut r, &mut w).unwrap(), 4);

    let mut r = repeat(0).take(1 << 17);
    assert_eq!(copy(&mut r as &mut dyn Read, &mut w as &mut dyn Write).unwrap(), 1 << 17);
}

struct ShortReader {
    cap: usize,
    read_size: usize,
    observed_buffer: usize,
}

impl Read for ShortReader {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let bytes = min(self.cap, self.read_size);
        self.cap -= bytes;
        self.observed_buffer = max(self.observed_buffer, buf.len());
        Ok(bytes)
    }
}

struct WriteObserver {
    observed_buffer: usize,
}

impl Write for WriteObserver {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        self.observed_buffer = max(self.observed_buffer, buf.len());
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn copy_specializes_bufwriter() {
    let cap = 117 * 1024;
    let buf_sz = 16 * 1024;
    let mut r = ShortReader { cap, observed_buffer: 0, read_size: 1337 };
    let mut w = BufWriter::with_capacity(buf_sz, WriteObserver { observed_buffer: 0 });
    assert_eq!(
        copy(&mut r, &mut w).unwrap(),
        cap as u64,
        "expected the whole capacity to be copied"
    );
    assert_eq!(r.observed_buffer, buf_sz, "expected a large buffer to be provided to the reader");
    assert!(w.get_mut().observed_buffer > DEFAULT_BUF_SIZE, "expected coalesced writes");
}

#[test]
fn sink_sinks() {
    let mut s = sink();
    assert_eq!(s.write(&[]).unwrap(), 0);
    assert_eq!(s.write(&[0]).unwrap(), 1);
    assert_eq!(s.write(&[0; 1024]).unwrap(), 1024);
    assert_eq!(s.by_ref().write(&[0; 1024]).unwrap(), 1024);
}

#[test]
fn empty_reads() {
    let mut e = empty();
    assert_eq!(e.read(&mut []).unwrap(), 0);
    assert_eq!(e.read(&mut [0]).unwrap(), 0);
    assert_eq!(e.read(&mut [0; 1024]).unwrap(), 0);
    assert_eq!(e.by_ref().read(&mut [0; 1024]).unwrap(), 0);

    let buf: &mut [MaybeUninit<_>] = &mut [];
    let mut buf: BorrowedBuf<'_> = buf.into();
    e.read_buf(buf.unfilled()).unwrap();
    assert_eq!(buf.len(), 0);
    assert_eq!(buf.init_len(), 0);

    let buf: &mut [_] = &mut [MaybeUninit::uninit()];
    let mut buf: BorrowedBuf<'_> = buf.into();
    e.read_buf(buf.unfilled()).unwrap();
    assert_eq!(buf.len(), 0);
    assert_eq!(buf.init_len(), 0);

    let buf: &mut [_] = &mut [MaybeUninit::uninit(); 1024];
    let mut buf: BorrowedBuf<'_> = buf.into();
    e.read_buf(buf.unfilled()).unwrap();
    assert_eq!(buf.len(), 0);
    assert_eq!(buf.init_len(), 0);

    let buf: &mut [_] = &mut [MaybeUninit::uninit(); 1024];
    let mut buf: BorrowedBuf<'_> = buf.into();
    e.by_ref().read_buf(buf.unfilled()).unwrap();
    assert_eq!(buf.len(), 0);
    assert_eq!(buf.init_len(), 0);
}

#[test]
fn empty_seeks() {
    let mut e = empty();
    assert!(matches!(e.seek(SeekFrom::Start(0)), Ok(0)));
    assert!(matches!(e.seek(SeekFrom::Start(1)), Ok(0)));
    assert!(matches!(e.seek(SeekFrom::Start(u64::MAX)), Ok(0)));

    assert!(matches!(e.seek(SeekFrom::End(i64::MIN)), Ok(0)));
    assert!(matches!(e.seek(SeekFrom::End(-1)), Ok(0)));
    assert!(matches!(e.seek(SeekFrom::End(0)), Ok(0)));
    assert!(matches!(e.seek(SeekFrom::End(1)), Ok(0)));
    assert!(matches!(e.seek(SeekFrom::End(i64::MAX)), Ok(0)));

    assert!(matches!(e.seek(SeekFrom::Current(i64::MIN)), Ok(0)));
    assert!(matches!(e.seek(SeekFrom::Current(-1)), Ok(0)));
    assert!(matches!(e.seek(SeekFrom::Current(0)), Ok(0)));
    assert!(matches!(e.seek(SeekFrom::Current(1)), Ok(0)));
    assert!(matches!(e.seek(SeekFrom::Current(i64::MAX)), Ok(0)));
}

#[test]
fn repeat_repeats() {
    let mut r = repeat(4);
    let mut b = [0; 1024];
    assert_eq!(r.read(&mut b).unwrap(), 1024);
    assert!(b.iter().all(|b| *b == 4));
}

#[test]
fn take_some_bytes() {
    assert_eq!(repeat(4).take(100).bytes().count(), 100);
    assert_eq!(repeat(4).take(100).bytes().next().unwrap().unwrap(), 4);
    assert_eq!(repeat(1).take(10).chain(repeat(2).take(10)).bytes().count(), 20);
}

#[allow(dead_code)]
fn const_utils() {
    const _: Empty = empty();
    const _: Repeat = repeat(b'c');
    const _: Sink = sink();
}
