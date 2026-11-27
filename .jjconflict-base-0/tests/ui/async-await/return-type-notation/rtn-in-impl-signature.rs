#![feature(return_type_notation)]

// Shouldn't ICE when we have a (bad) RTN in an impl header

trait Super1<'a> {
    fn bar<'b>() -> bool;
}

impl Super1<'_, bar(..): Send> for () {}
//~^ ERROR associated item constraints are not allowed here
//~| ERROR not all trait items implemented

fn main() {}
