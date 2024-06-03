trait Trait: Sized {}

impl dyn Trait { //~ ERROR the trait `Trait` cannot be made into an object
    fn function(&self) {} //~ ERROR the trait `Trait` cannot be made into an object
}

impl Trait for () {}

fn main() {
    <dyn Trait>::function(&());
    //~^ ERROR the trait `Trait` cannot be made into an object
    //~| ERROR the trait `Trait` cannot be made into an object
}
