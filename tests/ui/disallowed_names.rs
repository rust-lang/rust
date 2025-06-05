//@aux-build:proc_macros.rs
#![allow(
    dead_code,
    clippy::needless_if,
    clippy::similar_names,
    clippy::single_match,
    clippy::toplevel_ref_arg,
    unused_mut,
    unused_variables
)]
#![warn(clippy::disallowed_names)]

extern crate proc_macros;
use proc_macros::{external, with_span};

fn test(foo: ()) {}
//~^ disallowed_names

fn main() {
    let foo = 42;
    //~^ disallowed_names

    let baz = 42;
    //~^ disallowed_names

    let quux = 42;
    //~^ disallowed_names

    // Unlike these others, `bar` is actually considered an acceptable name.
    // Among many other legitimate uses, bar commonly refers to a period of time in music.
    // See https://github.com/rust-lang/rust-clippy/issues/5225.
    let bar = 42;

    let food = 42;
    let foodstuffs = 42;
    let bazaar = 42;

    match (42, Some(1337), Some(0)) {
        (foo, Some(baz), quux @ Some(_)) => (),
        //~^ disallowed_names
        //~| disallowed_names
        //~| disallowed_names
        _ => (),
    }
}

fn issue_1647(mut foo: u8) {
    //~^ disallowed_names

    let mut baz = 0;
    //~^ disallowed_names

    if let Some(mut quux) = Some(42) {}
    //~^ disallowed_names
}

fn issue_1647_ref() {
    let ref baz = 0;
    //~^ disallowed_names

    if let Some(ref quux) = Some(42) {}
    //~^ disallowed_names
}

fn issue_1647_ref_mut() {
    let ref mut baz = 0;
    //~^ disallowed_names

    if let Some(ref mut quux) = Some(42) {}
    //~^ disallowed_names
}

pub fn issue_14958_proc_macro() {
    // does not lint macro-generated code
    external! {
        let foo = 0;
    }
    with_span! {
        span
        let foo = 0;
    }
}

#[cfg(test)]
mod tests {
    fn issue_7305() {
        // `disallowed_names` lint should not be triggered inside of the test code.
        let foo = 0;

        // Check that even in nested functions warning is still not triggered.
        fn nested() {
            let foo = 0;
        }
    }
}

#[test]
fn test_with_disallowed_name() {
    let foo = 0;
}
