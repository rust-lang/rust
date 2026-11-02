use crate::sys::pal::error;
use crate::{fmt, io};

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

pub fn format_error(errno: i32, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if let Some(name) = error::error_name(errno) { f.write_str(name) } else { write!(f, "{errno}") }
}
