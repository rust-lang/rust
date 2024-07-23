//@ aux-crate:noprelude:somedep=somedep.rs
//@ compile-flags: -Zunstable-options
//@ edition:2018

fn main() {
    somedep::somefun();  //~ ERROR cannot find item
}
