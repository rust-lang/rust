#![feature(reborrow)]

use std::marker::Reborrow;

struct Thing<'a: 'static> {
    field: &'a mut u8,

}

impl<'a: 'static> Reborrow for Thing<'a> {}

fn takes(_: Thing<'_>) {}

fn main() {
    let mut x = 0;
    let thing = Thing { field: &mut x }; //~ ERROR does not live long enough
    takes({ thing });
}
