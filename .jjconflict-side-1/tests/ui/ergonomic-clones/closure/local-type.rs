// Check that using the parameter name in its type does not ICE.

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

fn main() {
    let _ = use |x: x| x; //~ ERROR cannot find type `x` in this scope
    let _ = use |x: bool| -> x { x }; //~ ERROR cannot find type `x` in this scope
}
