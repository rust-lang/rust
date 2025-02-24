//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius_next polonius
//@ [polonius_next] check-pass
//@ [polonius_next] compile-flags: -Zpolonius=next
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Zpolonius

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
            //[nll]~^ ERROR cannot borrow `**other` as mutable more than once at a time [E0499]
        };

        let c = other;
    }
}

fn main() {}
