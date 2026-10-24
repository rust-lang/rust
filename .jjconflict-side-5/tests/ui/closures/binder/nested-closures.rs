//@ check-pass

#![feature(closure_lifetime_binder)]

fn main() {
    for<'a> || -> () { for<'c> |_: &'a ()| -> () {}; };
}
