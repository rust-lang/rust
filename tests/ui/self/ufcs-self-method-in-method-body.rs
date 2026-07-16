//! Regression test for <https://github.com/rust-lang/rust/issues/25279>.
//! Ensure we're not ICE'ing on calls to Self method in method body.
//! (when implementing on type with lifetimes)
//@ run-pass

struct S<'a>(&'a ());

impl<'a> S<'a> {
    fn foo(self) -> &'a () {
        <Self>::bar(self)
    }

    fn bar(self) -> &'a () {
        self.0
    }
}

fn main() {
    S(&()).foo();
}
