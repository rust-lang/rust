// check-pass
// edition:2018

// This test is derived from
// https://github.com/rust-lang/rust/issues/72651#issuecomment-668720468

// This test demonstrates that, in `async fn g()`,
// indeed a temporary borrow `y` from `x` is live
// while `f().await` is being evaluated.
// Thus, `&'_ u8` should be included in type signature
// of the underlying generator.

async fn f() -> u8 { 1 }

pub async fn g(x: u8) {
    match x {
        y if f().await == y => (),
        _ => (),
    }
}

fn main() {
    let _ = g(10);
}
