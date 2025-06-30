trait Trait: Sized {}

impl dyn Trait { //~ ERROR the trait `Trait` is not dyn compatible
    fn function(&self) {} //~ ERROR the trait `Trait` is not dyn compatible
}

impl Trait for () {}

fn main() {
    <dyn Trait>::function(&());
    //~^ ERROR the trait `Trait` is not dyn compatible
    //~| ERROR the trait `Trait` is not dyn compatible
}
