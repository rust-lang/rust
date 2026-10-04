//! Regression test for https://github.com/rust-lang/rust/issues/106138.
//! From comment https://github.com/rust-lang/rust/pull/107567#discussion_r1093589448
//! Unary operator inference must not assume that the associated output type equals `Self`.

#[derive(Copy, Clone, Default)]
struct A;

struct B;

impl std::ops::Not for A {
    type Output = B;

    fn not(self) -> B {
        B
    }
}

#[derive(Copy, Clone)]
struct NoNot;

fn make<T>() -> T {
    loop {}
}

fn resolved_without_impl() {
    let value = make();
    let _ = !value;
    //~^ ERROR the trait bound `NoNot: Not` is not satisfied
    let _: NoNot = value;
}

fn unconstrained() {
    let value = make();
    //~^ ERROR type annotations needed
    let _ = !value;
}

fn main() {
    let x = Default::default();
    //~^ ERROR cannot call associated function on trait
    let _: A = !x;
}
