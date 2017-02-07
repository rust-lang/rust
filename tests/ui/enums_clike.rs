// ignore-x86
#![feature(plugin, associated_consts)]
#![plugin(clippy)]
#![deny(clippy)]

#![allow(unused)]

#[repr(usize)]
enum NonPortable {
    X = 0x1_0000_0000, //~ ERROR: Clike enum variant discriminant is not portable to 32-bit targets
    Y = 0,
    Z = 0x7FFF_FFFF,
    A = 0xFFFF_FFFF,
}

enum NonPortableNoHint {
    X = 0x1_0000_0000, //~ ERROR: Clike enum variant discriminant is not portable to 32-bit targets
    Y = 0,
    Z = 0x7FFF_FFFF,
    A = 0xFFFF_FFFF, //~ ERROR: Clike enum variant discriminant is not portable to 32-bit targets
}

#[repr(isize)]
enum NonPortableSigned {
    X = -1,
    Y = 0x7FFF_FFFF,
    Z = 0xFFFF_FFFF, //~ ERROR: Clike enum variant discriminant is not portable to 32-bit targets
    A = 0x1_0000_0000, //~ ERROR: Clike enum variant discriminant is not portable to 32-bit targets
    B = std::i32::MIN as isize,
    C = (std::i32::MIN as isize) - 1, //~ ERROR: Clike enum variant discriminant is not portable to 32-bit targets
}

enum NonPortableSignedNoHint {
    X = -1,
    Y = 0x7FFF_FFFF,
    Z = 0xFFFF_FFFF, //~ ERROR: Clike enum variant discriminant is not portable to 32-bit targets
    A = 0x1_0000_0000, //~ ERROR: Clike enum variant discriminant is not portable to 32-bit targets
}

/*
FIXME: uncomment once https://github.com/rust-lang/rust/issues/31910 is fixed
#[repr(usize)]
enum NonPortable2<T: Trait> {
    X = Trait::Number,
    Y = 0,
}

trait Trait {
    const Number: usize = 0x1_0000_0000;
}
*/

fn main() {
}
