use crate::io::prelude::*;
use crate::io::{BorrowedBuf, Empty, Repeat, SeekFrom, Sink, empty, repeat, sink};
use crate::mem::MaybeUninit;

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
    assert_eq!(Read::by_ref(&mut e).read(&mut [0; 1024]).unwrap(), 0);

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
    Read::by_ref(&mut e).read_buf(buf.unfilled()).unwrap();
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
fn empty_sinks() {
    let mut e = empty();
    assert_eq!(e.write(&[]).unwrap(), 0);
    assert_eq!(e.write(&[0]).unwrap(), 1);
    assert_eq!(e.write(&[0; 1024]).unwrap(), 1024);
    assert_eq!(Write::by_ref(&mut e).write(&[0; 1024]).unwrap(), 1024);
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
