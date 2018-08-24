// This assumes the packed and non-packed structs are different sizes.

// the error points to the start of the file, not the line with the
// transmute

// normalize-stderr-test "\d+ bits" -> "N bits"
// error-pattern: transmute called with types of different sizes

use std::mem;

#[repr(packed)]
struct Foo {
    bar: u8,
    baz: usize
}

#[derive(Debug)]
struct Oof {
    rab: u8,
    zab: usize
}

fn main() {
    let foo = Foo { bar: 1, baz: 10 };
    unsafe {
        let oof: Oof = mem::transmute(foo);
        println!("{:?}", oof);
    }
}
