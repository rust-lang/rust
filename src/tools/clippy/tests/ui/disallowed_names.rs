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

fn test(foo: ()) {}

fn main() {
    let foo = 42;
    let baz = 42;
    let quux = 42;
    // Unlike these others, `bar` is actually considered an acceptable name.
    // Among many other legitimate uses, bar commonly refers to a period of time in music.
    // See https://github.com/rust-lang/rust-clippy/issues/5225.
    let bar = 42;

    let food = 42;
    let foodstuffs = 42;
    let bazaar = 42;

    match (42, Some(1337), Some(0)) {
        (foo, Some(baz), quux @ Some(_)) => (),
        _ => (),
    }
}

fn issue_1647(mut foo: u8) {
    let mut baz = 0;
    if let Some(mut quux) = Some(42) {}
}

fn issue_1647_ref() {
    let ref baz = 0;
    if let Some(ref quux) = Some(42) {}
}

fn issue_1647_ref_mut() {
    let ref mut baz = 0;
    if let Some(ref mut quux) = Some(42) {}
}

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
