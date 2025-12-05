#![feature(closure_lifetime_binder)]
fn main() {
    for<> |_: &'a ()| -> () {};
    //~^ ERROR use of undeclared lifetime name `'a`
    for<'a> |_: &'b ()| -> () {};
    //~^ ERROR use of undeclared lifetime name `'b`
}
