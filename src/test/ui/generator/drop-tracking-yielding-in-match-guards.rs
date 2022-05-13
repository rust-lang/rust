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
    let _ = static |x: u8| match x {
        y if { yield } == y + 1 => (),
        _ => (),
    };
}
