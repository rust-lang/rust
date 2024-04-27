//@ aux-crate:somedep=somedep.rs
//@ compile-flags: -Zunstable-options -Dunused-crate-dependencies
//@ edition:2018

fn main() { //~ ERROR external crate `somedep` unused in `no_nounused`
}
