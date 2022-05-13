use super::BorrowBuf;
use crate::mem::MaybeUninit;

/// Test that BorrowBuf has the correct numbers when created with new
#[test]
fn new() {
    let buf: &mut [_] = &mut [0; 16];
    let mut rbuf: BorrowBuf<'_> = buf.into();

    assert_eq!(rbuf.filled().len(), 0);
    assert_eq!(rbuf.init_len(), 16);
    assert_eq!(rbuf.capacity(), 16);
    assert_eq!(rbuf.unfilled().capacity(), 16);
}

/// Test that BorrowBuf has the correct numbers when created with uninit
#[test]
fn uninit() {
    let buf: &mut [_] = &mut [MaybeUninit::uninit(); 16];
    let mut rbuf: BorrowBuf<'_> = buf.into();

    assert_eq!(rbuf.filled().len(), 0);
    assert_eq!(rbuf.init_len(), 0);
    assert_eq!(rbuf.capacity(), 16);
    assert_eq!(rbuf.unfilled().capacity(), 16);
}

#[test]
fn initialize_unfilled() {
    let buf: &mut [_] = &mut [MaybeUninit::uninit(); 16];
    let mut rbuf: BorrowBuf<'_> = buf.into();

    rbuf.unfilled().ensure_init();

    assert_eq!(rbuf.init_len(), 16);
}

#[test]
fn add_filled() {
    let buf: &mut [_] = &mut [0; 16];
    let mut rbuf: BorrowBuf<'_> = buf.into();

    unsafe {
        rbuf.unfilled().advance(1);
    }

    assert_eq!(rbuf.filled().len(), 1);
    assert_eq!(rbuf.unfilled().capacity(), 15);
}

#[test]
fn clear() {
    let buf: &mut [_] = &mut [255; 16];
    let mut rbuf: BorrowBuf<'_> = buf.into();

    unsafe {
        rbuf.unfilled().advance(16);
    }

    assert_eq!(rbuf.filled().len(), 16);
    assert_eq!(rbuf.unfilled().capacity(), 0);

    rbuf.clear();

    assert_eq!(rbuf.filled().len(), 0);
    assert_eq!(rbuf.unfilled().capacity(), 16);

    assert_eq!(rbuf.unfilled().init_ref(), [255; 16]);
}

#[test]
fn set_init() {
    let buf: &mut [_] = &mut [MaybeUninit::uninit(); 16];
    let mut rbuf: BorrowBuf<'_> = buf.into();

    unsafe {
        rbuf.set_init(8);
    }

    assert_eq!(rbuf.init_len(), 8);

    unsafe {
        rbuf.unfilled().advance(4);
    }

    unsafe {
        rbuf.set_init(2);
    }

    assert_eq!(rbuf.init_len(), 8);

    unsafe {
        rbuf.set_init(8);
    }

    assert_eq!(rbuf.init_len(), 8);
}

#[test]
fn append() {
    let buf: &mut [_] = &mut [MaybeUninit::new(255); 16];
    let mut rbuf: BorrowBuf<'_> = buf.into();

    rbuf.unfilled().append(&[0; 8]);

    assert_eq!(rbuf.init_len(), 8);
    assert_eq!(rbuf.filled().len(), 8);
    assert_eq!(rbuf.filled(), [0; 8]);

    rbuf.clear();

    rbuf.unfilled().append(&[1; 16]);

    assert_eq!(rbuf.init_len(), 16);
    assert_eq!(rbuf.filled().len(), 16);
    assert_eq!(rbuf.filled(), [1; 16]);
}
