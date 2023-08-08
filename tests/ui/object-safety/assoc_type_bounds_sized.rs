//! This test checks that associated types only need to be
//! mentioned in trait objects, if they don't require `Self: Sized`.

// check-pass

trait Foo {
    type Bar
    where
        Self: Sized;
}

fn foo(_: &dyn Foo) {}

trait Other: Sized {}

trait Boo {
    type Assoc
    where
        Self: Other;
}

fn boo(_: &dyn Boo) {}

fn main() {}
