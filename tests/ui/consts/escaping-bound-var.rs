#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete

fn test<'a>(
    _: &'a (),
) -> [(); { //~ ERROR: mismatched types
    let x: &'a ();
    //~^ ERROR cannot capture late-bound lifetime in constant
    1
}] {
}

fn main() {}
