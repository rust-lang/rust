//xfail-test

trait Cat {
    fn meow() -> bool;
    fn scratch() -> bool;
    fn purr() -> bool { true }
}

impl int : Cat {
    fn meow() -> bool {
        self.scratch()
    }
    fn scratch() -> bool {
        self.purr()
    }
}

fn main() {
    assert 5.meow();
}
