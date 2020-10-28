// Regression test for issue #78398
// Tests that we don't ICE when trying to print the AST json
// when we have capturd tokens for a foreign item

// check-pass
// compile-flags: -Zast-json

fn main() {}

macro_rules! mac_extern {
    ($i:item) => {
        extern "C" { $i }
    }
}

mac_extern! {
    fn foo();
}
