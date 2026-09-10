use std::fs::File;
use std::io;

#[derive(Debug)]
pub struct Lock(());

impl Lock {
    pub fn try_lock(_f: File, _exclusive: bool) -> io::Result<Lock> {
        let msg = "file locks not supported on this platform";
        Err(io::Error::new(io::ErrorKind::Unsupported, msg))
    }
}
