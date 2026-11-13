use core::mem::ManuallyDrop;

use super::{FileDesc, max_iov};
use crate::io::IoSlice;
use crate::os::unix::io::FromRawFd;

#[test]
fn limit_vector_count() {
    const IOV_MAX: usize = max_iov();

    let stdout = ManuallyDrop::new(unsafe { FileDesc::from_raw_fd(1) });
    let mut bufs = vec![IoSlice::new(&[]); IOV_MAX * 2 + 1];
    assert_eq!(stdout.write_vectored(&bufs).unwrap(), 0);

    // The slice of buffers is truncated to IOV_MAX buffers. However, since the
    // first IOV_MAX buffers are all empty, it is sliced starting at the first
    // non-empty buffer to avoid erroneously returning Ok(0). In this case, that
    // starts with the b"hello" buffer and ends just before the b"world!"
    // buffer.
    bufs[IOV_MAX] = IoSlice::new(b"hello");
    bufs[IOV_MAX * 2] = IoSlice::new(b"world!");
    assert_eq!(stdout.write_vectored(&bufs).unwrap(), b"hello".len())
}

#[test]
fn empty_vector() {
    let stdin = ManuallyDrop::new(unsafe { FileDesc::from_raw_fd(0) });
    assert_eq!(stdin.read_vectored(&mut []).unwrap(), 0);

    let stdout = ManuallyDrop::new(unsafe { FileDesc::from_raw_fd(1) });
    assert_eq!(stdout.write_vectored(&[]).unwrap(), 0);
}
