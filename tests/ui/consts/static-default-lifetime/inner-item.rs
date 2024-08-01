//@ check-pass

struct Foo<'a>(&'a ());

impl<'a> Foo<'a> {
    fn hello(self) {
        const INNER: &str = "";
    }
}

impl Foo<'_> {
    fn implicit(self) {
        const INNER: &str = "";
    }

    fn fn_lifetime(&self) {
        const INNER: &str = "";
    }
}

fn main() {}
