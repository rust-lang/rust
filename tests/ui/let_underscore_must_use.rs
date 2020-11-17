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

struct S {}

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
    let _ = g();
    let _ = h();
    let _ = l(0_u32);

    let s = S {};

    let _ = s.f();
    let _ = s.g();
    let _ = s.k();

    let _ = S::h();
    let _ = S::p();

    let _ = S::a();

    let _ = if true { Ok(()) } else { Err(()) };

    let a = Result::<(), ()>::Ok(());

    let _ = a.is_ok();

    let _ = a.map(|_| ());

    let _ = a;

    #[allow(clippy::let_underscore_must_use)]
    let _ = a;
}
