use crate::io;
use crate::path::{Path, PathBuf};
use crate::sys::unsupported;

pub(crate) fn absolute(_path: &Path) -> io::Result<PathBuf> {
    unsupported()
}
