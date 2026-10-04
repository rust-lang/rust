#![warn(clippy::struct_field_names)]

struct Data {
    a_data: u8,
    b_data: u8,
    c_data: u8,
    d_data: u8,
}
struct Data2 {
    //~^ ERROR: all fields have the same postfix
    a_data: u8,
    b_data: u8,
    c_data: u8,
    d_data: u8,
    e_data: u8,
}
enum Foo {
    AFoo,
    BFoo,
    CFoo,
    DFoo,
}
enum Foo2 {
    //~^ ERROR: all variants have the same postfix
    AFoo,
    BFoo,
    CFoo,
    DFoo,
    EFoo,
}

fn main() {}
