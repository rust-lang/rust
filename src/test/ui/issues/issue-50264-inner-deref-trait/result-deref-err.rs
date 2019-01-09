#![feature(inner_deref)]

fn main() {
    let _result = &Err(41).deref_err();
//~^ ERROR no method named `deref_err` found
}
