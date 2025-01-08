//@ edition:2021
//@ check-pass

#![feature(closure_lifetime_binder)]
#![feature(ergonomic_clones)]

fn main() {
    for<'a> use || -> () {};
}
