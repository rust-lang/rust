// Crate that exports a const fn. Used for testing cross-crate.

#![feature(const_fn)]
#![crate_type="rlib"]

pub const fn foo() -> usize { 22 }

pub const fn bar() -> fn() {
    fn x() {}
    x
}

#[inline]
pub const fn bar_inlined() -> fn() {
    fn x() {}
    x
}

#[inline(always)]
pub const fn bar_inlined_always() -> fn() {
    fn x() {}
    x
}
