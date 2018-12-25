#![feature(inner_deref)]

fn main() {
    let _result = &Ok(42).deref_ok();
//~^ ERROR no method named `deref_ok` found
}
