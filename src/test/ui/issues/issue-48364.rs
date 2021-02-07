fn foo() -> bool {
    b"".starts_with(stringify!(foo))
    //~^ ERROR arguments to this function are incorrect
}

fn main() {}
