#![warn(rust_2018_idioms)]

use bytes::buf::Buf;
use bytes::Bytes;

#[test]
fn long_take() {
    // Tests that get a take with a size greater than the buffer length will not
    // overrun the buffer. Regression test for #138.
    let buf = b"hello world".take(100);
    assert_eq!(11, buf.remaining());
    assert_eq!(b"hello world", buf.chunk());
}

#[test]
fn take_copy_to_bytes() {
    let mut abcd = Bytes::copy_from_slice(b"abcd");
    let abcd_ptr = abcd.as_ptr();
    let mut take = (&mut abcd).take(2);
    let a = take.copy_to_bytes(1);
    assert_eq!(Bytes::copy_from_slice(b"a"), a);
    // assert `to_bytes` did not allocate
    assert_eq!(abcd_ptr, a.as_ptr());
    assert_eq!(Bytes::copy_from_slice(b"bcd"), abcd);
}

#[test]
#[should_panic]
fn take_copy_to_bytes_panics() {
    let abcd = Bytes::copy_from_slice(b"abcd");
    abcd.take(2).copy_to_bytes(3);
}
