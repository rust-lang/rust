//xfail-test

// Right now, this fails with "attempted access of field `purr` on
// type `self`, but no public field or method with that name was
// found".

trait Cat {
    fn meow() -> bool;
    fn scratch() -> bool { self.purr() }
    fn purr() -> bool { true }
}

impl int : Cat {
    fn meow() -> bool {
        self.scratch()
    }
}

fn main() {
    assert 5.meow();
}
