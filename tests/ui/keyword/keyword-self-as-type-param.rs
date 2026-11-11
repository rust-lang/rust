// Regression test of #36638.

struct Foo<Self>(Self);
//~^ ERROR unexpected keyword `Self` in generic parameters
//~| ERROR recursive type `Foo` has infinite size

trait Bar<Self> {}
//~^ ERROR unexpected keyword `Self` in generic parameters

fn main() {}
