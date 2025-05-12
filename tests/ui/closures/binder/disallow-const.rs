#![feature(closure_lifetime_binder)]

fn main() {
    for<const N: i32> || -> () {};
    //~^ ERROR only lifetime parameters can be used in this context
}
