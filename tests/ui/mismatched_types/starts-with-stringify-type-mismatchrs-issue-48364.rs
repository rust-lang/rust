fn foo() -> bool {
    b"".starts_with(stringify!(foo))
    //~^ ERROR mismatched types
}

fn main() {}

// https://github.com/rust-lang/rust/issues/48364
