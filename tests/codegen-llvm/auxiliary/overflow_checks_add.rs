//@ compile-flags: -Cdebug-assertions=yes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

/// Emulates the default behavior of `+` using `intrinsics::overflow_checks()`.
#[inline]
pub fn add(a: u8, b: u8) -> u8 {
    if core::intrinsics::overflow_checks() { a.strict_add(b) } else { a.wrapping_add(b) }
}
