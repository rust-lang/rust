use crate::io;
use crate::sys::pal::error;

pub fn errno() -> i32 {
    0
}

#[inline]
pub fn is_interrupted(code: i32) -> bool {
    crate::sys::net::is_interrupted(code)
}

pub fn decode_error_kind(code: i32) -> io::ErrorKind {
    error::decode_error_kind(code)
}

pub fn error_string(errno: i32) -> String {
    if let Some(name) = error::error_name(errno) { name.to_owned() } else { format!("{errno}") }
}
