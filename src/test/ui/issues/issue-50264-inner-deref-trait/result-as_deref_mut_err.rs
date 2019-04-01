#![feature(inner_deref)]

fn main() {
    let _result = &mut Err(41).as_deref_mut_err();
//~^ ERROR no method named `as_deref_mut_err` found
}
