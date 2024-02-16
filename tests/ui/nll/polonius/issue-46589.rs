// This test is a copy of `ui/nll/issue-46589.rs` which fails in NLL but succeeds in Polonius.
// As we can't have a test here which conditionally passes depending on a test
// revision/compile-flags. We ensure here that it passes in Polonius mode.

//@ check-pass
//@ compile-flags: -Z polonius

struct Foo;

impl Foo {
    fn get_self(&mut self) -> Option<&mut Self> {
        Some(self)
    }

    fn new_self(&mut self) -> &mut Self {
        self
    }

    fn trigger_bug(&mut self) {
        let other = &mut (&mut *self);

        *other = match (*other).get_self() {
            Some(s) => s,
            None => (*other).new_self()
        };

        let c = other;
    }
}

fn main() {}
