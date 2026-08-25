pub fn errno() -> i32 {
    0
}

pub fn is_interrupted(_code: i32) -> bool {
    false
}

pub fn decode_error_kind(_code: i32) -> crate::io::ErrorKind {
    crate::io::ErrorKind::Uncategorized
}

pub fn error_string(_errno: i32) -> String {
    "operation successful".to_string()
}
