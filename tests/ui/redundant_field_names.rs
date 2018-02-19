#![warn(redundant_field_names)]
#![allow(unused_variables)]

mod foo {
    pub const BAR: u8 = 0;
}

struct Person {
    gender: u8,
    age: u8,
    name: u8,
    buzz: u64,
    foo: u8,
}

fn main() {
    let gender: u8 = 42;
    let age = 0;
    let fizz: u64 = 0;
    let name: u8 = 0;

    let me = Person {
        gender: gender,
        age: age,

        name, //should be ok
        buzz: fizz, //should be ok
        foo: foo::BAR, //should be ok
    };
}
