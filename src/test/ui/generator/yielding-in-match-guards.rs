// check-pass
// edition:2018

// This test is derived from
// https://github.com/rust-lang/rust/issues/74961#issuecomment-666893845
// by @SNCPlay42

// This test demonstrates that, in `async fn g()`,
// indeed a temporary borrow `y` from `x` is live
// while `f().await` is being evaluated.
// Thus, `&'_ A` should be included in type signature
// of the underlying generator.

#[derive(PartialEq, Eq)]
struct A;

async fn f() -> A {
    A
}

async fn g() {
    let x = A;
    match x {
        y if f().await == y => {}
        _ => {}
    }
}

fn main() {}