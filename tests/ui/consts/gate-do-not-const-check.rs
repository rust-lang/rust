#[rustc_do_not_const_check]
//~^ ERROR this is an internal attribute that will never be stable
const fn foo() {}

fn main() {}
