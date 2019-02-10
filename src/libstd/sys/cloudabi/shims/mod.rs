use crate::io;

pub mod args;
pub mod env;
pub mod fs;
pub mod net;
#[path = "../../unix/path.rs"]
pub mod path;
pub mod pipe;
pub mod process;
pub mod os;

// This enum is used as the storage for a bunch of types which can't actually exist.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Void {}

pub fn unsupported<T>() -> io::Result<T> {
    Err(io::Error::new(
        io::ErrorKind::Other,
        "This function is not available on CloudABI.",
    ))
}
