// Test that we use fully-qualified type names in error messages.

mod x {
    pub enum foo { }
}

mod y {
    pub enum foo { }
}

fn bar(x: x::foo) -> y::foo {
    return x;
    //~^ ERROR mismatched types
    //~| expected type `y::foo`
    //~| found type `x::foo`
    //~| expected enum `y::foo`, found enum `x::foo`
}

fn main() {
}
