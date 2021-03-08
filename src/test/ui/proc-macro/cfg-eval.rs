// check-pass
// compile-flags: -Z span-debug
// aux-build:test-macros.rs

#![feature(cfg_eval)]
#![feature(proc_macro_hygiene)]
#![feature(stmt_expr_attributes)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

#[cfg_eval]
#[print_attr]
struct S1 {
    #[cfg(FALSE)]
    field_false: u8,
    #[cfg(all(/*true*/))]
    #[cfg_attr(FALSE, unknown_attr)]
    #[cfg_attr(all(/*true*/), allow())]
    field_true: u8,
}

#[cfg_eval]
#[cfg(FALSE)]
struct S2 {}

fn main() {
    let _ = #[cfg_eval] #[print_attr](#[cfg(FALSE)] 0, #[cfg(all(/*true*/))] 1);
}
