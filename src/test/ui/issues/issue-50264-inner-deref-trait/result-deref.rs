#![feature(inner_deref)]

fn main() {
    let _result = &Ok(42).deref();
//~^ ERROR no method named `deref` found
}
