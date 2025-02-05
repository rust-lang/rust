//@ run-pass

trait Trait {
    fn function(&self) {}
}

impl dyn Trait {
}

impl Trait for () {}

fn main() {
    Trait::function(&());
}
