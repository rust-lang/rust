use crate::fmt;
use crate::os::xous::ffi::Error as XousError;

pub fn errno() -> i32 {
    0
}

pub fn is_interrupted(_code: i32) -> bool {
    false
}

pub fn decode_error_kind(_code: i32) -> crate::io::ErrorKind {
    crate::io::ErrorKind::Uncategorized
}

pub fn format_error(errno: i32, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let error = XousError::from(errno);
    write!(f, "{error}")
}
