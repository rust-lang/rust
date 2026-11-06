// This checks whether a single fn called from two trivial contexts produces a
// combined diagnostic for both of them

#![feature(rustc_attrs)]
#![allow(internal_features)]
#![deny(dead_code)] //~ NOTE: the lint level is defined here

struct Struct { //~ NOTE: field in this struct
    value: u32, //~ ERROR: `value` is never read
}

fn read(data: &Struct) { //~ ERROR: function `read` is never used
    let _ = data.value;
}

#[rustc_trivial_field_reads]
fn a(data: &Struct) {
    //~^ NOTE: `value` is used here, but this is intentionally ignored
    //~| NOTE: `read` is used here, but this is intentionally ignored
    read(data);
}

#[rustc_trivial_field_reads]
fn b(data: &Struct) {
    //~^ NOTE: `value` is used here, but this is intentionally ignored
    //~| NOTE: `read` is used here, but this is intentionally ignored
    read(data);
}

fn main() {
    let data = Struct { value: 42 };
    a(&data);
    b(&data);
}
