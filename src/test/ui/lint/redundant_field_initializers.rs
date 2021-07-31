// run-rustfix
#![deny(redundant_field_initializers)]
#![allow(dead_code, unused_variables)]

use std::ops::{Range, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive};

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
        //~^ ERROR redundant field names
        age: age,
        //~^ ERROR redundant field names

        name,          //should be ok
        buzz: fizz,    //should be ok
        foo: foo::BAR, //should be ok
    };

    // Range expressions
    let (start, end) = (0, 0);

    let _ = start..;
    let _ = ..end;
    let _ = start..end;

    let _ = ..=end;
    let _ = start..=end;

    // Issue #2799
    let _: Vec<_> = (start..end).collect();

    // hand-written Range family structs are linted
    let _ = RangeFrom { start: start }; //~ ERROR redundant field names
    let _ = RangeTo { end: end }; //~ ERROR redundant field names
    let _ = Range {
        start: start, //~ ERROR redundant field names
        end: end //~ ERROR redundant field names
    };
    let _ = RangeInclusive::new(start, end);
    let _ = RangeToInclusive { end: end }; //~ ERROR redundant field names
}

fn issue_3476() {
    fn foo<T>() {}

    struct S {
        foo: fn(),
    }

    S { foo: foo::<i32> };
}
