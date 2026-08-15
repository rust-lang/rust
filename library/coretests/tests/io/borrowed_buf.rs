use core::io::BorrowedBuf;
use core::mem::MaybeUninit;

/// Test that BorrowedBuf has the correct numbers when created with new
#[test]
fn new() {
    let buf: &mut [_] = &mut [0; 16];
    let mut rbuf: BorrowedBuf<'_, u8> = buf.into();

    assert_eq!(rbuf.filled().len(), 0);
    assert!(rbuf.is_init());
    assert_eq!(rbuf.capacity(), 16);
    assert_eq!(rbuf.unfilled().capacity(), 16);
}

#[test]
fn new_u64() {
    let buf: &mut [_] = &mut [0u64; 16];
    let mut rbuf = BorrowedBuf::from(buf);

    assert_eq!(rbuf.filled().len(), 0);
    assert!(rbuf.is_init());
    assert_eq!(rbuf.capacity(), 16);
    assert_eq!(rbuf.unfilled().capacity(), 16);
}

#[test]
fn new_unit() {
    let buf: &mut [_] = &mut [(); 16];
    let mut rbuf = BorrowedBuf::from(buf);

    assert_eq!(rbuf.filled().len(), 0);
    assert!(rbuf.is_init());
    assert_eq!(rbuf.capacity(), 16);
    assert_eq!(rbuf.unfilled().capacity(), 16);
}

/// Test that BorrowedBuf has the correct numbers when created with uninit
#[test]
fn uninit() {
    let buf: &mut [_] = &mut [MaybeUninit::uninit(); 16];
    let mut rbuf: BorrowedBuf<'_, u8> = buf.into();

    assert_eq!(rbuf.filled().len(), 0);
    assert!(!rbuf.is_init());
    assert_eq!(rbuf.capacity(), 16);
    assert_eq!(rbuf.unfilled().capacity(), 16);
}

#[test]
fn uninit_u64() {
    let buf: &mut [_] = &mut [MaybeUninit::<u64>::uninit(); 16];
    let mut rbuf = BorrowedBuf::<u64>::from(buf);

    assert_eq!(rbuf.filled().len(), 0);
    assert!(!rbuf.is_init());
    assert_eq!(rbuf.capacity(), 16);
    assert_eq!(rbuf.unfilled().capacity(), 16);
}

#[test]
fn uninit_unit() {
    let buf: &mut [_] = &mut [MaybeUninit::<()>::uninit(); 16];
    let mut rbuf = BorrowedBuf::<()>::from(buf);

    assert_eq!(rbuf.filled().len(), 0);
    assert!(!rbuf.is_init());
    assert_eq!(rbuf.capacity(), 16);
    assert_eq!(rbuf.unfilled().capacity(), 16);
}

#[test]
fn initialize_unfilled() {
    let buf: &mut [_] = &mut [MaybeUninit::uninit(); 16];
    let mut rbuf: BorrowedBuf<'_, u8> = buf.into();

    rbuf.unfilled().ensure_init();

    assert!(rbuf.is_init());
}

#[test]
fn advance_filled() {
    let buf: &mut [_] = &mut [0; 16];
    let mut rbuf: BorrowedBuf<'_, u8> = buf.into();

    rbuf.unfilled().advance_checked(1);

    assert_eq!(rbuf.filled().len(), 1);
    assert_eq!(rbuf.unfilled().capacity(), 15);
}

#[test]
fn advance_filled_u64() {
    let buf: &mut [_] = &mut [0u64; 16];
    let mut rbuf = BorrowedBuf::from(buf);

    rbuf.unfilled().advance_checked(1);

    assert_eq!(rbuf.filled().len(), 1);
    assert_eq!(rbuf.unfilled().capacity(), 15);
}

#[test]
fn advance_filled_unit() {
    let buf: &mut [_] = &mut [(); 16];
    let mut rbuf = BorrowedBuf::from(buf);

    rbuf.unfilled().advance_checked(1);

    assert_eq!(rbuf.filled().len(), 1);
    assert_eq!(rbuf.unfilled().capacity(), 15);
}

#[test]
fn clear() {
    let buf: &mut [_] = &mut [255; 16];
    let mut rbuf: BorrowedBuf<'_, u8> = buf.into();

    rbuf.unfilled().advance_checked(16);

    assert_eq!(rbuf.filled().len(), 16);
    assert_eq!(rbuf.unfilled().capacity(), 0);

    rbuf.clear();

    assert_eq!(rbuf.filled().len(), 0);
    assert_eq!(rbuf.unfilled().capacity(), 16);

    assert_eq!(rbuf.unfilled().ensure_init(), [255; 16]);
}

#[test]
fn clear_u64() {
    let buf: &mut [_] = &mut [255u64; 16];
    let mut rbuf = BorrowedBuf::from(buf);

    rbuf.unfilled().advance_checked(16);

    assert_eq!(rbuf.filled().len(), 16);
    assert_eq!(rbuf.unfilled().capacity(), 0);

    rbuf.clear();

    assert_eq!(rbuf.filled().len(), 0);
    assert_eq!(rbuf.unfilled().capacity(), 16);

    unsafe {
        rbuf.set_init();
    }
    rbuf.unfilled().advance_checked(16);
    assert_eq!(rbuf.filled(), [255; 16]);
}

#[test]
fn clear_unit() {
    let buf: &mut [_] = &mut [(); 16];
    let mut rbuf = BorrowedBuf::from(buf);

    rbuf.unfilled().advance_checked(16);

    assert_eq!(rbuf.filled().len(), 16);
    assert_eq!(rbuf.unfilled().capacity(), 0);

    rbuf.clear();

    assert_eq!(rbuf.filled().len(), 0);
    assert_eq!(rbuf.unfilled().capacity(), 16);

    unsafe {
        rbuf.set_init();
    }
    rbuf.unfilled().advance_checked(16);
    assert_eq!(rbuf.filled(), [(); 16]);
}

#[test]
fn set_init() {
    let buf: &mut [_] = &mut [MaybeUninit::zeroed(); 16];
    let mut rbuf: BorrowedBuf<'_, u8> = buf.into();

    unsafe {
        rbuf.set_init();
    }

    assert!(rbuf.is_init());
    rbuf.unfilled().advance_checked(16);
    assert_eq!(rbuf.filled(), [0; 16]);
}

#[test]
fn set_init_u64() {
    let buf: &mut [_] = &mut [MaybeUninit::<u64>::zeroed(); 16];
    let mut rbuf = BorrowedBuf::<u64>::from(buf);

    unsafe {
        rbuf.set_init();
    }

    assert!(rbuf.is_init());
    rbuf.unfilled().advance_checked(16);
    assert_eq!(rbuf.filled(), [0; 16]);
}

#[test]
fn set_init_unit() {
    let buf: &mut [_] = &mut [MaybeUninit::<()>::zeroed(); 16];
    let mut rbuf = BorrowedBuf::<()>::from(buf);

    unsafe {
        rbuf.set_init();
    }

    assert!(rbuf.is_init());
    rbuf.unfilled().advance_checked(16);
    assert_eq!(rbuf.filled(), [(); 16]);
}

#[test]
fn append() {
    let buf: &mut [_] = &mut [MaybeUninit::new(255); 16];
    let mut rbuf: BorrowedBuf<'_, u8> = buf.into();

    rbuf.unfilled().append(&[0; 8]);

    assert!(!rbuf.is_init());
    assert_eq!(rbuf.filled().len(), 8);
    assert_eq!(rbuf.filled(), [0; 8]);

    rbuf.clear();

    rbuf.unfilled().append(&[1; 16]);

    assert!(!rbuf.is_init());
    assert_eq!(rbuf.filled().len(), 16);
    assert_eq!(rbuf.filled(), [1; 16]);
}

#[test]
fn append_u64() {
    let buf: &mut [_] = &mut [MaybeUninit::new(255u64); 16];
    let mut rbuf = BorrowedBuf::<u64>::from(buf);

    rbuf.unfilled().append(&[0; 8]);

    assert!(!rbuf.is_init());
    assert_eq!(rbuf.filled().len(), 8);
    assert_eq!(rbuf.filled(), [0; 8]);

    rbuf.clear();

    rbuf.unfilled().append(&[1; 16]);

    assert!(!rbuf.is_init());
    assert_eq!(rbuf.filled().len(), 16);
    assert_eq!(rbuf.filled(), [1; 16]);
}

#[test]
fn append_unit() {
    let buf: &mut [_] = &mut [MaybeUninit::new(()); 16];
    let mut rbuf = BorrowedBuf::<()>::from(buf);

    rbuf.unfilled().append(&[(); 8]);

    assert!(!rbuf.is_init());
    assert_eq!(rbuf.filled().len(), 8);
    assert_eq!(rbuf.filled(), [(); 8]);

    rbuf.clear();

    rbuf.unfilled().append(&[(); 16]);

    assert!(!rbuf.is_init());
    assert_eq!(rbuf.filled().len(), 16);
    assert_eq!(rbuf.filled(), [(); 16]);
}

#[test]
fn reborrow_written() {
    let buf: &mut [_] = &mut [MaybeUninit::new(0); 32];
    let mut buf: BorrowedBuf<'_, u8> = buf.into();

    let mut cursor = buf.unfilled();
    cursor.append(&[1; 16]);

    let mut cursor2 = cursor.reborrow();
    cursor2.append(&[2; 16]);

    assert_eq!(cursor2.written(), 32);
    assert_eq!(cursor.written(), 32);

    assert_eq!(buf.unfilled().written(), 32);
    assert!(!buf.is_init());
    assert_eq!(buf.filled().len(), 32);
    let filled = buf.filled();
    assert_eq!(&filled[..16], [1; 16]);
    assert_eq!(&filled[16..], [2; 16]);
}

#[test]
fn reborrow_written_u64() {
    let buf: &mut [_] = &mut [MaybeUninit::new(0u64); 32];
    let mut buf = BorrowedBuf::<u64>::from(buf);

    let mut cursor = buf.unfilled();
    cursor.append(&[1; 16]);

    let mut cursor2 = cursor.reborrow();
    cursor2.append(&[2; 16]);

    assert_eq!(cursor2.written(), 32);
    assert_eq!(cursor.written(), 32);

    assert_eq!(buf.unfilled().written(), 32);
    assert!(!buf.is_init());
    assert_eq!(buf.filled().len(), 32);
    let filled = buf.filled();
    assert_eq!(&filled[..16], [1; 16]);
    assert_eq!(&filled[16..], [2; 16]);
}

#[test]
fn reborrow_written_unit() {
    let buf: &mut [_] = &mut [MaybeUninit::new(()); 32];
    let mut buf = BorrowedBuf::<()>::from(buf);

    let mut cursor = buf.unfilled();
    cursor.append(&[(); 16]);

    let mut cursor2 = cursor.reborrow();
    cursor2.append(&[(); 16]);

    assert_eq!(cursor2.written(), 32);
    assert_eq!(cursor.written(), 32);

    assert_eq!(buf.unfilled().written(), 32);
    assert!(!buf.is_init());
    assert_eq!(buf.filled().len(), 32);
    let filled = buf.filled();
    assert_eq!(&filled[..16], [(); 16]);
    assert_eq!(&filled[16..], [(); 16]);
}

#[test]
fn cursor_set_init() {
    let buf: &mut [_] = &mut [MaybeUninit::zeroed(); 16];
    let mut rbuf: BorrowedBuf<'_, u8> = buf.into();
    let mut cursor = rbuf.unfilled();

    unsafe {
        cursor.set_init();
    }

    assert!(cursor.is_init());
    assert_eq!(unsafe { cursor.as_mut().len() }, 16);

    cursor.advance_checked(4);

    assert_eq!(unsafe { cursor.as_mut().len() }, 12);

    assert!(rbuf.is_init());
}

#[test]
fn cursor_set_init_u64() {
    let buf: &mut [_] = &mut [MaybeUninit::<u64>::zeroed(); 16];
    let mut rbuf = BorrowedBuf::<u64>::from(buf);
    let mut cursor = rbuf.unfilled();

    unsafe {
        cursor.set_init();
    }

    assert!(cursor.is_init());
    assert_eq!(unsafe { cursor.as_mut().len() }, 16);

    cursor.advance_checked(4);

    assert_eq!(unsafe { cursor.as_mut().len() }, 12);

    assert!(rbuf.is_init());
}

#[test]
fn cursor_set_init_unit() {
    let buf: &mut [_] = &mut [MaybeUninit::<()>::zeroed(); 16];
    let mut rbuf = BorrowedBuf::<()>::from(buf);
    let mut cursor = rbuf.unfilled();

    unsafe {
        cursor.set_init();
    }

    assert!(cursor.is_init());
    assert_eq!(unsafe { cursor.as_mut().len() }, 16);

    cursor.advance_checked(4);

    assert_eq!(unsafe { cursor.as_mut().len() }, 12);

    assert!(rbuf.is_init());
}

#[test]
fn cursor_with_unfilled_buf() {
    let buf: &mut [_] = &mut [MaybeUninit::uninit(); 16];
    let mut rbuf = BorrowedBuf::from(buf);
    let mut cursor = rbuf.unfilled();

    cursor.with_unfilled_buf(|buf| {
        buf.unfilled().append(&[1, 2, 3]);
        assert_eq!(buf.filled(), &[1, 2, 3]);
    });

    assert!(!cursor.is_init());
    assert_eq!(cursor.written(), 3);

    cursor.with_unfilled_buf(|buf| {
        assert_eq!(buf.capacity(), 13);
        assert!(!buf.is_init());

        buf.unfilled().ensure_init();
        buf.unfilled().advance_checked(4);
    });

    assert!(cursor.is_init());
    assert_eq!(cursor.written(), 7);

    cursor.with_unfilled_buf(|buf| {
        assert_eq!(buf.capacity(), 9);
        assert!(buf.is_init());
    });

    assert!(cursor.is_init());
    assert_eq!(cursor.written(), 7);

    assert_eq!(rbuf.filled(), &[1, 2, 3, 0, 0, 0, 0]);
}
