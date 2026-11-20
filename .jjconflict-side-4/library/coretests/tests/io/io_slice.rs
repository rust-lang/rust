use core::io::{IoSlice, IoSliceMut};
use core::ops::Deref;

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
