//@ check-pass

#![feature(rustc_attrs)]
#![allow(internal_features)]
#![deny(dead_code)]

struct Struct {
    value: u32,
}

fn read(d: &Struct) {
    let _ = d.value;
}

#[rustc_trivial_field_reads]
fn trivial(d: &Struct) {
    read(d);
}

#[allow(dead_code)]
fn example() {
    let d = Struct { value: 42 };
    read(&d);
    trivial(&d);
}

fn main() {}
