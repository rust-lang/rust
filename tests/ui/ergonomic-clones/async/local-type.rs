// Check that using the parameter name in its type does not ICE.
//@ edition:2018

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

fn main() {
    let _ = async use |x: x| x; //~ ERROR cannot find type `x` in this scope
    let _ = async use |x: bool| -> x { x }; //~ ERROR cannot find type `x` in this scope
}
