//! Ensure we catch calls to `core`.

#![no_std]

#[unsafe(no_mangle)]
pub fn call_from_core(s: &[u8]) -> &str {
    match core::str::from_utf8(&s) {
        Ok(s) => s,
        Err(_) => "",
    }
}
