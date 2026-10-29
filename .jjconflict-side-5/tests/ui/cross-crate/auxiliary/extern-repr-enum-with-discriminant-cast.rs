//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/42007
#[repr(u8)]
pub enum E {
    B = 1 as u8,
}
