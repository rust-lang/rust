trait Trait {
    fn function() {}
}

impl Trait for () {}

fn main() {
    Trait::function();
    //~^ ERROR cannot call associated function on trait without specifying the corresponding `impl` type
}
