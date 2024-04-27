#![feature(closure_lifetime_binder)]

fn main() {
    for<T> || -> () {};
    //~^ ERROR only lifetime parameters can be used in this context
}
