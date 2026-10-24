//@ edition:2021
//@ check-pass

#![feature(closure_lifetime_binder)]

fn main() {
    for<'a> async || -> () {};
}
