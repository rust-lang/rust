// This assumes the packed and non-packed structs are different sizes.

// the error points to the start of the file, not the line with the
// transmute

// error-pattern: cannot transmute between types of different sizes, or dependently-sized types

use std::mem;

#[repr(packed)]
struct Foo<T,S> {
    bar: T,
    baz: S
}

struct Oof<T, S> {
    rab: T,
    zab: S
}

fn main() {
    let foo = Foo { bar: [1u8, 2, 3, 4, 5], baz: 10i32 };
    unsafe {
        let oof: Oof<[u8; 5], i32> = mem::transmute(foo);
        println!("{:?} {:?}", &oof.rab[..], oof.zab);
    }
}
