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

async fn f() -> u8 { 1 }

async fn i(x: u8) {
    match x {
        y if f().await == y + 1 => (),
        _ => (),
    }
}

fn main() {
    let _ = i(8);
}
