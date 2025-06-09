#![feature(generic_const_items, trivial_bounds)]
#![allow(incomplete_features, dead_code, trivial_bounds)]

// FIXME(generic_const_items): This looks like a bug to me. I expected that we wouldn't emit any
// errors. I thought we'd skip the evaluation of consts whose bounds don't hold.

const UNUSED: () = ()
where
    String: Copy;
//~^^^ ERROR unreachable code

fn main() {}
