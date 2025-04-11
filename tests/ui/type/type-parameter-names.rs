// Test that we print out the names of type parameters correctly in
// our error messages.

fn foo<Foo, Bar>(x: Foo) -> Bar {
    x
//~^ ERROR mismatched types
//~| NOTE_NONVIRAL expected type parameter `Bar`, found type parameter `Foo`
//~| NOTE_NONVIRAL expected type parameter `Bar`
//~| NOTE_NONVIRAL found type parameter `Foo`
}

fn main() {}
