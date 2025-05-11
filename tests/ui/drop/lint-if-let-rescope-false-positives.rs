//@ edition: 2021
//@ check-pass

#![deny(if_let_rescope)]

struct Drop;
impl std::ops::Drop for Drop {
    fn drop(&mut self) {
        println!("drop")
    }
}

impl Drop {
    fn as_ref(&self) -> Option<i32> {
        Some(1)
    }
}

fn consume(_: impl Sized) -> Option<i32> { Some(1) }

fn main() {
    let drop = Drop;

    // Make sure we don't drop if we don't actually make a temporary.
    if let None = drop.as_ref() {} else {}

    // Make sure we don't lint if we consume the droppy value.
    if let None = consume(Drop) {} else {}

    // Make sure we don't lint on field exprs of place exprs.
    let tup_place = (Drop, ());
    if let None = consume(tup_place.1) {} else {}
}
