#![warn(clippy::let_underscore_must_use)]
#![allow(clippy::unnecessary_wraps)]

// Debug implementations can fire this lint,
// so we shouldn't lint external macros
#[derive(Debug)]
struct Foo {
    field: i32,
}

#[must_use]
fn f() -> u32 {
    0
}

fn g() -> Result<u32, u32> {
    Ok(0)
}

#[must_use]
fn l<T>(x: T) -> T {
    x
}

fn h() -> u32 {
    0
}

struct S;

impl S {
    #[must_use]
    pub fn f(&self) -> u32 {
        0
    }

    pub fn g(&self) -> Result<u32, u32> {
        Ok(0)
    }

    fn k(&self) -> u32 {
        0
    }

    #[must_use]
    fn h() -> u32 {
        0
    }

    fn p() -> Result<u32, u32> {
        Ok(0)
    }
}

trait Trait {
    #[must_use]
    fn a() -> u32;
}

impl Trait for S {
    fn a() -> u32 {
        0
    }
}

fn main() {
    let _ = f();
    //~^ ERROR: non-binding `let` on a result of a `#[must_use]` function
    let _ = g();
    //~^ ERROR: non-binding `let` on an expression with `#[must_use]` type
    let _ = h();
    let _ = l(0_u32);
    //~^ ERROR: non-binding `let` on a result of a `#[must_use]` function

    let s = S {};

    let _ = s.f();
    //~^ ERROR: non-binding `let` on a result of a `#[must_use]` function
    let _ = s.g();
    //~^ ERROR: non-binding `let` on an expression with `#[must_use]` type
    let _ = s.k();

    let _ = S::h();
    //~^ ERROR: non-binding `let` on a result of a `#[must_use]` function
    let _ = S::p();
    //~^ ERROR: non-binding `let` on an expression with `#[must_use]` type

    let _ = S::a();
    //~^ ERROR: non-binding `let` on a result of a `#[must_use]` function

    let _ = if true { Ok(()) } else { Err(()) };
    //~^ ERROR: non-binding `let` on an expression with `#[must_use]` type

    let a = Result::<(), ()>::Ok(());

    let _ = a.is_ok();
    //~^ ERROR: non-binding `let` on a result of a `#[must_use]` function

    let _ = a.map(|_| ());
    //~^ ERROR: non-binding `let` on an expression with `#[must_use]` type

    let _ = a;
    //~^ ERROR: non-binding `let` on an expression with `#[must_use]` type

    #[allow(clippy::let_underscore_must_use)]
    let _ = a;
}
