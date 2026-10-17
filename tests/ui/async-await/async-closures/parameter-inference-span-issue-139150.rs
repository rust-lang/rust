//! Regression test for <https://github.com/rust-lang/rust/issues/139150>.
// Prefer the source parameter over the lowered async closure body for type annotations.

//@ edition: 2024

fn wildcard() {
    async |_| {};
    //~^ ERROR type annotations needed
}

fn named() {
    let _ = async |value| { let _ = value; };
    //~^ ERROR type annotations needed
}

fn tuple() {
    let _ = async |(first, second)| { let _ = (first, second); };
    //~^ ERROR type annotations needed
}

fn by_ref() {
    let _ = async |ref value| { let _ = value; };
    //~^ ERROR type annotations needed
}

fn explicit() {
    let _ = async |_: Option<_>| {};
    //~^ ERROR type annotations needed
}

fn synchronous() {
    let _ = |_| {};
    //~^ ERROR type annotations needed
}

macro_rules! generated {
    () => { async |_| {} };
    //~^ ERROR type annotations needed
}

fn from_macro() {
    let _ = generated!();
}

fn main() {}
