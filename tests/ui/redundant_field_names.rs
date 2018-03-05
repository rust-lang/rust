#![warn(redundant_field_names)]
#![allow(unused_variables)]
#![feature(inclusive_range,inclusive_range_syntax)]

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

    // Range syntax
    let (start, end) = (0, 0);

    let _ = start..;
    let _ = ..end;
    let _ = start..end;

    let _ = ..=end;
    let _ = start..=end;

    // TODO: the following should be linted
    let _ = ::std::ops::RangeFrom { start: start };
    let _ = ::std::ops::RangeTo { end: end };
    let _ = ::std::ops::Range { start: start, end: end };
    let _ = ::std::ops::RangeInclusive { start: start, end: end };
    let _ = ::std::ops::RangeToInclusive { end: end };
}
