// Test that we print out the names of type parameters correctly in
// our error messages.

fn foo<Foo, Bar>(x: Foo) -> Bar {
    x
//~^ ERROR mismatched types
//~| expected type `Bar`
//~| found type `Foo`
//~| expected type parameter, found a different type parameter
}

fn main() {}
