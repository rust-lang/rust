// Test a cycle where a type parameter on a trait has a default that
// again references the trait.

trait Foo<X = Box<dyn Foo>> {
    //~^ ERROR cycle detected
}

fn main() { }
