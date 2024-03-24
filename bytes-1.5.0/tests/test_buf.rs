#![warn(rust_2018_idioms)]

use bytes::Buf;
#[cfg(feature = "std")]
use std::io::IoSlice;

#[test]
fn test_fresh_cursor_vec() {
    let mut buf = &b"hello"[..];

    assert_eq!(buf.remaining(), 5);
    assert_eq!(buf.chunk(), b"hello");

    buf.advance(2);

    assert_eq!(buf.remaining(), 3);
    assert_eq!(buf.chunk(), b"llo");

    buf.advance(3);

    assert_eq!(buf.remaining(), 0);
    assert_eq!(buf.chunk(), b"");
}

#[test]
fn test_get_u8() {
    let mut buf = &b"\x21zomg"[..];
    assert_eq!(0x21, buf.get_u8());
}

#[test]
fn test_get_u16() {
    let mut buf = &b"\x21\x54zomg"[..];
    assert_eq!(0x2154, buf.get_u16());
    let mut buf = &b"\x21\x54zomg"[..];
    assert_eq!(0x5421, buf.get_u16_le());
}

#[test]
#[should_panic]
fn test_get_u16_buffer_underflow() {
    let mut buf = &b"\x21"[..];
    buf.get_u16();
}

#[cfg(feature = "std")]
#[test]
fn test_bufs_vec() {
    let buf = &b"hello world"[..];

    let b1: &[u8] = &mut [];
    let b2: &[u8] = &mut [];

    let mut dst = [IoSlice::new(b1), IoSlice::new(b2)];

    assert_eq!(1, buf.chunks_vectored(&mut dst[..]));
}

#[test]
fn test_vec_deque() {
    use std::collections::VecDeque;

    let mut buffer: VecDeque<u8> = VecDeque::new();
    buffer.extend(b"hello world");
    assert_eq!(11, buffer.remaining());
    assert_eq!(b"hello world", buffer.chunk());
    buffer.advance(6);
    assert_eq!(b"world", buffer.chunk());
    buffer.extend(b" piece");
    let mut out = [0; 11];
    buffer.copy_to_slice(&mut out);
    assert_eq!(b"world piece", &out[..]);
}

#[allow(unused_allocation)] // This is intentional.
#[test]
fn test_deref_buf_forwards() {
    struct Special;

    impl Buf for Special {
        fn remaining(&self) -> usize {
            unreachable!("remaining");
        }

        fn chunk(&self) -> &[u8] {
            unreachable!("chunk");
        }

        fn advance(&mut self, _: usize) {
            unreachable!("advance");
        }

        fn get_u8(&mut self) -> u8 {
            // specialized!
            b'x'
        }
    }

    // these should all use the specialized method
    assert_eq!(Special.get_u8(), b'x');
    assert_eq!((&mut Special as &mut dyn Buf).get_u8(), b'x');
    assert_eq!((Box::new(Special) as Box<dyn Buf>).get_u8(), b'x');
    assert_eq!(Box::new(Special).get_u8(), b'x');
}

#[test]
fn copy_to_bytes_less() {
    let mut buf = &b"hello world"[..];

    let bytes = buf.copy_to_bytes(5);
    assert_eq!(bytes, &b"hello"[..]);
    assert_eq!(buf, &b" world"[..])
}

#[test]
#[should_panic]
fn copy_to_bytes_overflow() {
    let mut buf = &b"hello world"[..];

    let _bytes = buf.copy_to_bytes(12);
}
