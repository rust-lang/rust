// Test that we print out the names of type parameters correctly in
// our error messages.

fn foo<Foo, Bar>(x: Foo) -> Bar {
    x
//~^ ERROR mismatched types
//~| expected type parameter `Bar`, found type parameter `Foo`
//~| expected type parameter `Bar`
//~| found type parameter `Foo`
}

fn main() {}
