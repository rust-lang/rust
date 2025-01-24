trait Trait {
    fn baz(&self, _: Self) {}
    //~^ ERROR the size for values of type `Self` cannot be known
    fn bat(&self) -> Self {}
    //~^ ERROR mismatched types
    //~| ERROR the size for values of type `Self` cannot be known
}

fn bar(x: &dyn Trait) {} //~ ERROR the trait `Trait` is not dyn compatible

trait Other: Sized {}

fn foo(x: &dyn Other) {} //~ ERROR the trait `Other` is not dyn compatible

fn main() {}
