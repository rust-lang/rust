//@ build-pass
//@ edition:2018

// This test is derived from
// https://github.com/rust-lang/rust/issues/72651#issuecomment-668720468

// This test demonstrates that, in `async fn g()`,
// indeed a temporary borrow `y` from `x` is live
// while `f().await` is being evaluated.
// Thus, `&'_ u8` should be included in type signature
// of the underlying coroutine.

#![feature(if_let_guard)]

async fn f() -> u8 { 1 }
async fn foo() -> [bool; 10] { [false; 10] }

pub async fn g(x: u8) {
    match x {
        y if f().await == y => (),
        _ => (),
    }
}

// #78366: check the reference to the binding is recorded even if the binding is not autorefed

async fn h(x: usize) {
    match x {
        y if foo().await[y] => (),
        _ => (),
    }
}

async fn i(x: u8) {
    match x {
        y if f().await == y + 1 => (),
        _ => (),
    }
}

async fn j(x: u8) {
    match x {
        y if let (1, 42) = (f().await, y) => (),
        _ => (),
    }
}

fn main() {
    let _ = g(10);
    let _ = h(9);
    let _ = i(8);
    let _ = j(7);
}
