#![feature(closure_lifetime_binder)]

fn main() {
    let _f = for<'a> |_: &'a ()| {};
    //~^ implicit return type is forbidden when `for<...>` is present
}
