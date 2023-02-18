#![feature(closure_lifetime_binder, non_lifetime_binders)]
//~^ WARN  is incomplete and may not be safe to use

fn main()  {
    for<T> || -> T {};
    //~^ ERROR late-bound type parameter not allowed on closures
}
