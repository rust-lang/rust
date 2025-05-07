//@ proc-macro: test-macros.rs

// Regression test for issue #81543
// Tests that we emit a properly spanned error
// when the output of a proc-macro cannot be parsed
// as the expected AST node kind

extern crate test_macros;

test_macros::identity! {
    fn 32() {} //~ ERROR expected identifier
}

fn main() {}
