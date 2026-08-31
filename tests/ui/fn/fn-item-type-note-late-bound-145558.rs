//! Regression test for https://github.com/rust-lang/rust/issues/145558
//!
//! The note explaining that distinct fn items have distinct types was suppressed when the
//! signatures contained a late-bound lifetime, because the two binders name their bound
//! region differently.

//@ dont-require-annotations: NOTE

struct A;

fn f1<'a>(_: &'a A) {}
fn f2<'a>(_: &'a A) {}

fn main() {
    let mut map = vec![];
    map.push(f1);
    map.push(f2);
    //~^ ERROR mismatched types
    //~| NOTE different fn items have unique types
}
