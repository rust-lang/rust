//! Check that unstable name-resolution suggestions are omitted on stable.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/149402>.
//!
//@ needs-target-std
//@ ignore-backends: gcc

//@ revisions: stable nightly
//@[stable] act-as-stable
//@[nightly] only-nightly

fn foo() {
    let x = Vec::new();
    x.push(Complete::Item { name: "hello" });
    //~^ ERROR: cannot find type `Complete` in this scope
}

fn main() {}
