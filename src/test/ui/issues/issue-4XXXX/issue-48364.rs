fn foo() -> bool {
    b"".starts_with(stringify!(foo))
    //~^ ERROR mismatched types
}

fn main() {}
