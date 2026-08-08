//@ run-pass
//@ reference: layout.swift.struct-offsets
//@ edition: 2024

// LLVM <{ i64, i8 }>
#[repr(Swift)]
struct S {
    a: isize,
    b: u8,
}

// LLVM <{ i8, [7 x i8], <{ i64, i8 }>, i8, [6 x i8] i64, i64 }>
#[repr(Swift)]
struct S2 {
    a: u8,
    b: S,
    c: u8,
    d: isize,
    e: (),
    f: isize,
}

fn main() {
    assert_eq!(core::mem::offset_of!(S, a), 0);
    assert_eq!(core::mem::offset_of!(S, b), 8);

    assert_eq!(core::mem::offset_of!(S2, a), 0);
    assert_eq!(core::mem::offset_of!(S2, b), 8);
    assert_eq!(core::mem::offset_of!(S2, c), 17);
    assert_eq!(core::mem::offset_of!(S2, d), 24);
    assert_eq!(core::mem::offset_of!(S2, e), 32);
    assert_eq!(core::mem::offset_of!(S2, f), 32);

    assert_eq!(core::mem::size_of::<S2>(), 40);
    assert_eq!(core::mem::align_of::<S2>(), 8);
}
