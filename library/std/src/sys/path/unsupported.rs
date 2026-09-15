use crate::io;
use crate::path::Path;

#[inline]
pub fn with_native_path<T>(path: &Path, f: &dyn Fn(&Path) -> io::Result<T>) -> io::Result<T> {
    f(path)
}
