//@ revisions: gated active

#![cfg_attr(active, feature(pattern_types))]
#![allow(incomplete_features)]

// Check that pattern types do not affect existing macros.
// They don't, because pattern types don't have surface syntax.

macro_rules! foo {
    ($t:ty is $p:pat) => {}; //~ ERROR `$t:ty` is followed by `is`, which is not allowed for `ty` fragments
}

fn main() {
    foo!(u32 is 1..)
}
