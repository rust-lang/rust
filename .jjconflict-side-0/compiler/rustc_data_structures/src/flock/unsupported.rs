use std::fs::File;
use std::io;
use std::path::Path;

#[derive(Debug)]
pub struct Lock(());

impl Lock {
    pub fn try_lock(_p: &Path, _f: File, _exclusive: bool) -> io::Result<Lock> {
        let msg = "file locks not supported on this platform";
        Err(io::Error::new(io::ErrorKind::Unsupported, msg))
    }
}
