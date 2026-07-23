use crate::io;
use crate::path::Path;

pub fn with_native_path<T>(path: &Path, f: &dyn Fn(&str) -> io::Result<T>) -> io::Result<T> {
    let path = path.to_str().ok_or(io::Error::from(io::ErrorKind::InvalidFilename))?;
    f(path)
}
