//@ check-pass
//@ compile-flags: -Z span-debug
//@ proc-macro: test-macros.rs

#![feature(cfg_eval)]
#![feature(proc_macro_hygiene)]
#![feature(stmt_expr_attributes)]
#![feature(rustc_attrs)]
#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

#[cfg_eval]
#[print_attr]
struct S1 {
    #[cfg(false)]
    field_false: u8,
    #[cfg(all(/*true*/))]
    #[cfg_attr(FALSE, unknown_attr)]
    #[cfg_attr(all(/*true*/), allow())]
    field_true: u8,
}

#[cfg_eval]
#[cfg(false)]
struct S2 {}

fn main() {
    // Subtle - we need a trailing comma after the '1' - otherwise, `#[cfg_eval]` will
    // turn this into `(#[cfg(all())] 1)`, which is a parenthesized expression, not a tuple
    // expression. `#[cfg]` is not supported inside parenthesized expressions, so this will
    // produce an error when attribute collection runs.
    let _ = #[cfg_eval] #[print_attr] #[cfg_attr(not(FALSE), rustc_dummy)]
    (#[cfg(false)] 0, #[cfg(all(/*true*/))] 1,);
}
