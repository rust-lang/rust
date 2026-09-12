//! Regression test for https://github.com/rust-lang/rust/issues/160171.
//! A misplaced `pub` before a block's closing brace must not ICE.

fn main() {
    pub
    //~^ ERROR visibility `pub` is not followed by an item
} //~ ERROR expected expression, found `}`
