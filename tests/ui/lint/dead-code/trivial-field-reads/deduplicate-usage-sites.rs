// This test uses an item through a trivial helper twice
// from different seeds, to ensure the note pointing to
// the `rustc_trivial_field_reads` attr is added only once
#![feature(rustc_attrs)]
#![deny(dead_code)]

struct Data {
    value: bool, //~ ERROR: field `value` is never read
}

#[rustc_trivial_field_reads]
fn helper(data: &Data) {
    println!("{}", data.value);
}

#[rustc_trivial_field_reads]
fn intermediate(data: &Data) {
    helper(data);
}

#[allow(unused)]
fn allowed() {
    let data = Data { value: true };
    helper(&data);
}

fn main() {
    let data = Data { value: true };
    intermediate(&data);
}
