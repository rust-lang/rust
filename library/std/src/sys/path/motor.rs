use crate::io;
use crate::os::motor::ffi::OsStrExt;
use crate::path::Path;

#[inline]
pub fn with_native_path<T>(path: &Path, f: &dyn Fn(&str) -> io::Result<T>) -> io::Result<T> {
    f(path.as_str())
}
