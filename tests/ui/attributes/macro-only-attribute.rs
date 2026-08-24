//! `#[allow_internal_unstable]` and `#[allow_internal_unsafe]` may only be applied to macros.
//! Applying them to an ordinary (non-procedural-macro) function is an error.
#![feature(allow_internal_unstable)]
#![feature(allow_internal_unsafe)]

#[allow_internal_unstable(something)] //~ ERROR attribute should be applied to a macro
fn not_a_macro_unstable() {}

#[allow_internal_unsafe] //~ ERROR attribute should be applied to a macro
fn not_a_macro_unsafe() {}

// Applying them to a `macro_rules!` macro is fine.
#[allow_internal_unstable(something)]
#[allow_internal_unsafe]
macro_rules! ok_on_macro {
    () => {};
}

fn main() {}
