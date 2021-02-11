// edition:2021

trait Trait {}

trait X {
    fn foo(&self)
    //~^ ERROR the trait `X` cannot be made into an object
    where
        Self: Trait;
}

impl X for () {
    fn foo(&self) {}
}

impl Trait for dyn X {}

pub fn main() {
    <dyn X as X>::foo(&());
}
