// Regression test for https://github.com/rust-lang/rust/issues/165471

#![feature(gca_macroless_items)]
#![feature(gca_min_const_items)]

struct S;
trait A {
    const DEFAULT: Option<Self> = None::<()>;
    //~^ ERROR implementation of a regular const cannot have a `gca!` RHS
    //~| ERROR the type of const parameters must not depend on other generic parameters
}
impl A for S {}
fn f() {
    let _t: Option<S> = A::DEFAULT;
}

fn main() {}
