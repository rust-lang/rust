#![feature(inner_deref)]

fn main() {
    let _result = &Err(41).as_deref_err();
//~^ ERROR no method named `as_deref_err` found
}
