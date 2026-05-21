use crate::io::{IoSlice, Result};

#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn default_write_vectored<F>(write: F, bufs: &[IoSlice<'_>]) -> Result<usize>
where
    F: FnOnce(&[u8]) -> Result<usize>,
{
    let buf = bufs.iter().find(|b| !b.is_empty()).map_or(&[][..], |b| &**b);
    write(buf)
}
