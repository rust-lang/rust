//@ check-pass

#![feature(closure_lifetime_binder)]
#![feature(rustc_attrs)]

#[rustc_regions]
fn main() {
    for<'a> || -> () { for<'c> |_: &'a ()| -> () {}; };
}
