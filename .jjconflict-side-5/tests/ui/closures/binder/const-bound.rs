#![feature(closure_lifetime_binder, non_lifetime_binders)]

fn main()  {
    for<const N: i32> || -> () {};
    //~^ ERROR late-bound const parameters cannot be used currently
    //~| ERROR late-bound const parameter not allowed on closures
}
