//! Checks variations of E0057, which is the incorrect number of agruments passed into a closure

//@ check-fail

fn foo<T: Fn()>(t: T) {
    t(1i32);
    //~^ ERROR function takes 0 arguments but 1 argument was supplied
}

/// Regression test for <https://github.com/rust-lang/rust/issues/16939>
fn foo2<T: Fn()>(f: T) {
    |t| f(t);
    //~^ ERROR function takes 0 arguments but 1 argument was supplied
}

fn bar(t: impl Fn()) {
    t(1i32);
    //~^ ERROR function takes 0 arguments but 1 argument was supplied
}

fn baz() -> impl Fn() {
    || {}
}

fn baz2() {
    baz()(1i32)
    //~^ ERROR function takes 0 arguments but 1 argument was supplied
}

fn qux() {
    let x = || {};
    x(1i32);
    //~^ ERROR function takes 0 arguments but 1 argument was supplied
}

fn main() {}
