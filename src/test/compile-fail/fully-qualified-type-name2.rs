// Test that we use fully-qualified type names in error messages.

mod x {
    enum foo { }
}

mod y {
    enum foo { }
}

fn bar(x: x::foo) -> y::foo {
    ret x;
    //~^ ERROR mismatched types: expected `y::foo` but found `x::foo`
}

fn main() {
}
