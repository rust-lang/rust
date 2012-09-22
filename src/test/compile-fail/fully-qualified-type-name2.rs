// Test that we use fully-qualified type names in error messages.

mod x {
    #[legacy_exports];
    enum foo { }
}

mod y {
    #[legacy_exports];
    enum foo { }
}

fn bar(x: x::foo) -> y::foo {
    return x;
    //~^ ERROR mismatched types: expected `y::foo` but found `x::foo`
}

fn main() {
}
