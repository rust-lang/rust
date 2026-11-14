// Check that using the parameter name in its type does not ICE.

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

fn main() {
    let _ = |x: x| x.use; //~ ERROR cannot find type `x` in this scope
    let _ = |x: bool| -> x { x.use }; //~ ERROR cannot find type `x` in this scope
}
