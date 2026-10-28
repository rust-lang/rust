// Check that using the parameter name in its type does not ICE.
//@ edition:2018

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

fn main() {
    let _ = async |x: x| x.use; //~ ERROR cannot find type `x` in this scope
    let _ = async |x: bool| -> x { x.use }; //~ ERROR cannot find type `x` in this scope
}
