trait Mumbo {
    fn jumbo(&self, x: &usize) -> usize;
}

impl Mumbo for usize {
    // Cannot have a larger effect than the trait:
    unsafe fn jumbo(&self, x: &usize) { *self + *x; }
    //~^ ERROR method `jumbo` has an incompatible type for trait
    //~| expected type `fn
    //~| found type `unsafe fn
}

fn main() {}
