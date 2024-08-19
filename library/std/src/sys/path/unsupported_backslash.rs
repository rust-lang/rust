#![forbid(unsafe_op_in_unsafe_fn)]
use crate::io;
use crate::path::{Path, PathBuf};
use crate::sys::unsupported;

pub(crate) fn absolute(_path: &Path) -> io::Result<PathBuf> {
    unsupported()
}
