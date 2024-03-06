//@ edition:2021
//@ check-pass

#![feature(closure_lifetime_binder)]
#![feature(async_closure)]

fn main() {
    for<'a> async || -> () {};
}
