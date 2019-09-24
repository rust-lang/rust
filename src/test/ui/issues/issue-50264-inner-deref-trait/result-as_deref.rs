#![feature(inner_deref)]

fn main() {
    let _result = &Ok(42).as_deref();
//~^ ERROR no method named `as_deref` found
}
