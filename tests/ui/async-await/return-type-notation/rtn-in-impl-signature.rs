#![feature(return_type_notation)]
//~^ WARN the feature `return_type_notation` is incomplete

// Shouldn't ICE when we have a (bad) RTN in an impl header

trait Super1<'a> {
    fn bar<'b>() -> bool;
}

impl Super1<'_, bar(): Send> for () {}
//~^ ERROR associated type bindings are not allowed here

fn main() {}
