//xfail-test

// Currently failing with an ICE in trans.

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
