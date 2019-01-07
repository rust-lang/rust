// compile-pass

// Test that `extern crate self;` is accepted
// syntactically as an item for use in a macro.

macro_rules! accept_item { ($x:item) => {} }

accept_item! {
    extern crate self;
}

fn main() {}
