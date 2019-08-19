#![feature(inner_deref)]

fn main() {
    let _result = &Ok(42).as_deref_ok();
//~^ ERROR no method named `as_deref_ok` found
}
