// Test that we print out the names of type parameters correctly in
// our error messages.
// related issue <https://github.com/rust-lang/rust/issues/2951>

//@ dont-require-annotations: NOTE

fn foo<Foo, Bar>(x: Foo) -> Bar {
    x
    //~^ ERROR mismatched types
    //~| NOTE expected type parameter `Bar`, found type parameter `Foo`
    //~| NOTE expected type parameter `Bar`
    //~| NOTE found type parameter `Foo`
}

fn bar<Foo, Bar>(x: Foo, y: Bar) {
    let mut xx = x;
    xx = y;
    //~^ ERROR mismatched types
    //~| NOTE expected type parameter `Foo`, found type parameter `Bar`
    //~| NOTE expected type parameter `Foo`
    //~| NOTE found type parameter `Bar`
}

fn main() {}
