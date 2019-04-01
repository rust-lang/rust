fn foo() -> bool {
    b"".eq_ignore_ascii_case(stringify!(foo))
    //~^ ERROR mismatched types
}

fn main() {}
