//@ compile-flags: -Cdebug-assertions=yes

#![crate_type = "lib"]
#![feature(strict_overflow_ops)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]

/// Emulates the default behavior of `+` using
/// `intrinsics::overflow_checks()`.
#[inline]
#[rustc_inherit_overflow_checks]
pub fn add(a: u8, b: u8) -> u8 {
    if core::intrinsics::overflow_checks() { a.strict_add(b) } else { a.wrapping_add(b) }
}
