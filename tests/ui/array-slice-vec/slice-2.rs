// Test that slicing syntax gives errors if we have not implemented the trait.

struct Foo;

fn main() {
    let x = Foo;
    &x[..]; //~ ERROR cannot index into a value of type `Foo`
    &x[Foo..]; //~ ERROR cannot index into a value of type `Foo`
    &x[..Foo]; //~ ERROR cannot index into a value of type `Foo`
    &x[Foo..Foo]; //~ ERROR cannot index into a value of type `Foo`
}
