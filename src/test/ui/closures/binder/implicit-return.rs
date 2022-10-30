#![feature(closure_lifetime_binder)]

fn main() {
    let _f = for<'a> |_: &'a ()| {};
    //~^ implicit types in closure signatures are forbidden when `for<...>` is present
}
