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
//~^ ERROR: use of a disallowed/placeholder name `foo`
//~| NOTE: `-D clippy::disallowed-names` implied by `-D warnings`

fn main() {
    let foo = 42;
    //~^ ERROR: use of a disallowed/placeholder name `foo`
    let baz = 42;
    //~^ ERROR: use of a disallowed/placeholder name `baz`
    let quux = 42;
    //~^ ERROR: use of a disallowed/placeholder name `quux`
    // Unlike these others, `bar` is actually considered an acceptable name.
    // Among many other legitimate uses, bar commonly refers to a period of time in music.
    // See https://github.com/rust-lang/rust-clippy/issues/5225.
    let bar = 42;

    let food = 42;
    let foodstuffs = 42;
    let bazaar = 42;

    match (42, Some(1337), Some(0)) {
        (foo, Some(baz), quux @ Some(_)) => (),
        //~^ ERROR: use of a disallowed/placeholder name `foo`
        //~| ERROR: use of a disallowed/placeholder name `baz`
        //~| ERROR: use of a disallowed/placeholder name `quux`
        _ => (),
    }
}

fn issue_1647(mut foo: u8) {
    //~^ ERROR: use of a disallowed/placeholder name `foo`
    let mut baz = 0;
    //~^ ERROR: use of a disallowed/placeholder name `baz`
    if let Some(mut quux) = Some(42) {}
    //~^ ERROR: use of a disallowed/placeholder name `quux`
}

fn issue_1647_ref() {
    let ref baz = 0;
    //~^ ERROR: use of a disallowed/placeholder name `baz`
    if let Some(ref quux) = Some(42) {}
    //~^ ERROR: use of a disallowed/placeholder name `quux`
}

fn issue_1647_ref_mut() {
    let ref mut baz = 0;
    //~^ ERROR: use of a disallowed/placeholder name `baz`
    if let Some(ref mut quux) = Some(42) {}
    //~^ ERROR: use of a disallowed/placeholder name `quux`
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
