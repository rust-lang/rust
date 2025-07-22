//@compile-flags: -Zmin-function-alignment=8

// FIXME(rust-lang/rust#82232, rust-lang/rust#143834): temporarily renamed to mitigate `#[align]`
// nameres ambiguity
#![feature(rustc_attrs)]
#![feature(fn_align)]

// When a function uses `align(N)`, the function address should be a multiple of `N`.

#[rustc_align(256)]
fn foo() {}

#[rustc_align(16)]
fn bar() {}

#[rustc_align(4)]
fn baz() {}

fn main() {
    assert!((foo as usize).is_multiple_of(256));
    assert!((bar as usize).is_multiple_of(16));

    // The maximum of `align(N)` and `-Zmin-function-alignment=N` is used.
    assert!((baz as usize).is_multiple_of(8));
}
