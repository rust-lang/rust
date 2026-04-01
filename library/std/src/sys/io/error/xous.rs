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

pub fn error_string(errno: i32) -> String {
    Into::<XousError>::into(errno).to_string()
}
