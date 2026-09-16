use std::fs::File;
use std::io;

#[derive(Debug)]
pub struct Lock(());

impl Lock {
    pub fn new(_f: File, _wait: bool, _exclusive: bool) -> io::Result<Lock> {
        let msg = "file locks not supported on this platform";
        Err(io::Error::new(io::ErrorKind::Unsupported, msg))
    }
}
