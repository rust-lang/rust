use crate::io;
use crate::path::PathBuf;

pub fn getcwd() -> io::Result<PathBuf> {
    Ok(PathBuf::from("/"))
}

pub fn temp_dir() -> PathBuf {
    PathBuf::from("/tmp")
}
