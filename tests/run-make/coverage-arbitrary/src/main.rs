use core::hint::black_box;
use std::env::args;

use arbitrary::{Arbitrary, Unstructured};

#[derive(Debug, Arbitrary)]
struct MyStruct {
    _x: u32,
}

#[derive(Debug, Arbitrary)]
enum MyEnum {
    One,
    Two,
    Three,
}

fn main() {
    // Print the executable path to stdout, so that the rmake script has easy
    // access to it. This is easier than trying to interrogate cargo.
    println!("{}", args().nth(0).unwrap());

    dbg!(MyStruct::size_hint(0));
    dbg!(MyEnum::size_hint(0));

    let mut data = Unstructured::new(black_box(&[0; 1024]));

    dbg!(MyStruct::arbitrary(&mut data));
    dbg!(MyEnum::arbitrary(&mut data));
}
