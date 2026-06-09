//@ edition: 2021

// Regression test for #148542
// Ensure we don't ICE with "Invalid `ConstKind` for `const_to_pat`: {const error}"

fn foo() where &str:, {
    //~^ ERROR `&` without an explicit lifetime name cannot be used here
    match 42_u8 {
        -10.. => {}
    }
}

fn main() {}
