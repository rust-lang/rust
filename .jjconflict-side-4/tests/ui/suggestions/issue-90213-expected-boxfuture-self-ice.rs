// Checks that we do not ICE when comparing `Self` to `Pin`
//@ edition:2021

struct S;

impl S {
    fn foo(_: Box<Option<S>>) {}
    fn bar() {
        Self::foo(None) //~ ERROR mismatched types
    }
}

fn main() {}
