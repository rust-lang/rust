//! Regression test for <https://github.com/rust-lang/rust/issues/39687>
#![feature(fn_traits)]

fn main() {
    <fn() as Fn()>::call;
    //~^ ERROR associated item constraints are not allowed here [E0229]
}
