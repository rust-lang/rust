//@ edition:2015
//@ run-rustfix
//@ compile-flags: -Aunused

#[cfg(true)]
use y::Whatever;

mod y {
    pub(crate) fn z() {}
    pub(crate) struct Whatever;
}

fn main() {
    z();
    //~^ ERROR cannot find function `z` in this scope
}
