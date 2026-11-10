#![feature(rustc_attrs)]
#![deny(dead_code)] //~ NOTE: lint level is defined here
#![allow(internal_features)]

struct Struct { //~ NOTE: fields in this struct
    first: bool, //~ ERROR: fields `first` and `second` are never read
    second: bool,
}

#[rustc_trivial_field_reads]
fn read_first(data: &Struct) { //~ NOTE: `first` is used in `read_first`
    let _ = data.first;
}

#[rustc_trivial_field_reads]
fn read_second(data: &Struct) { //~ NOTE: `second` is used in `read_second`
    let _ = data.second;
}

#[rustc_trivial_field_reads]
fn read_first_and_second(data: &Struct) { //~ NOTE: `first` and `second` are used in `read_first_and_second`
    let _ = data.first;
    let _ = data.second;
}

fn main() {
    let data = Struct { first: false, second: false };
    read_first(&data);
    read_second(&data);
    read_first_and_second(&data);
}
