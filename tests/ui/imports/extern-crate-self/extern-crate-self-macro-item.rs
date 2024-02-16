//@ build-pass (FIXME(62277): could be check-pass?)

// Test that `extern crate self;` is accepted
// syntactically as an item for use in a macro.

macro_rules! accept_item { ($x:item) => {} }

accept_item! {
    extern crate self;
}

fn main() {}
