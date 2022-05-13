// build-pass
// edition:2018
// compile-flags: -Zdrop-tracking

// This test is derived from
// https://github.com/rust-lang/rust/issues/72651#issuecomment-668720468

// This test demonstrates that, in `async fn g()`,
// indeed a temporary borrow `y` from `x` is live
// while `f().await` is being evaluated.
// Thus, `&'_ u8` should be included in type signature
// of the underlying generator.

#![feature(generators)]

fn main() {
    let _a = static |x: u8| match x {
        y if { yield } == y + 1 => (),
        _ => (),
    };

    static STATIC: u8 = 42;
    let _b = static |x: u8| match x {
        y if { yield } == STATIC + 1 => (),
        _ => (),
    };

    let upvar = 42u8;
    let _c = static |x: u8| match x {
        y if { yield } == upvar + 1 => (),
        _ => (),
    };
}
