use core::mem::ManuallyDrop;

use super::{FileDesc, IoSlice};
use crate::os::unix::io::FromRawFd;

#[test]
fn limit_vector_count() {
    let stdout = ManuallyDrop::new(unsafe { FileDesc::from_raw_fd(1) });
    let bufs = (0..1500).map(|_| IoSlice::new(&[])).collect::<Vec<_>>();
    assert!(stdout.write_vectored(&bufs).is_ok());
}
