//! Regression test for issue #156713. In the `fails` case, borrowck was trying to check liveness
//! of `bar` which had been moved to a match scrutinee.
//@ edition: 2015
//@ check-pass

struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {}
}

struct Bar<'a>(&'a Foo);
impl Drop for Bar<'_> {
    fn drop(&mut self) {}
}

// This compiles
fn works() {
    let foo = Foo;
    let bar = Bar(&foo);
    drop(match { (bar,) } {
        args => args,
    })
}

// This used to error
fn fails() {
    let foo = Foo;
    let bar = Bar(&foo);
    drop(match (bar,) {
        args => args,
    })
}

fn main() {}
