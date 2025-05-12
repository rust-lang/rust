//@ aux-crate:somedep=somedep.rs
//@ compile-flags: -Zunstable-options -Dunused-crate-dependencies
//@ edition:2018

fn main() { //~ ERROR extern crate `somedep` is unused in crate `no_nounused`
}
