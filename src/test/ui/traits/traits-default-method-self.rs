// run-pass


trait Cat {
    fn meow(&self) -> bool;
    fn scratch(&self) -> bool { self.purr() }
    fn purr(&self) -> bool { true }
}

impl Cat for isize {
    fn meow(&self) -> bool {
        self.scratch()
    }
}

pub fn main() {
    assert!(5.meow());
}
