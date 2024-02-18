//@ edition:2015
//@ run-rustfix

#![allow(unused_variables)]
#![deny(keyword_idents)]

fn main() {
    let dyn = (); //~ ERROR dyn
    //~^ WARN this is accepted in the current edition
}
