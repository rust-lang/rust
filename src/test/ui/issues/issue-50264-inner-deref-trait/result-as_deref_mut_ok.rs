#![feature(inner_deref)]

fn main() {
    let _result = &mut Ok(42).as_deref_mut_ok();
//~^ ERROR no method named `as_deref_mut_ok` found
}
