#![feature(closure_lifetime_binder, non_lifetime_binders)]

fn main() {
    for<T> || -> () {};
    //~^ ERROR late-bound type parameter not allowed on closures
}
