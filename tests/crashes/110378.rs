//@ known-bug: #110378
// ignore-tidy-linelength

#![feature(generic_const_exprs)]

fn foo<const L: usize>(_a: [u8; L], _b: [u8; L]) -> [u8; L + 1] {
    [0_u8; L + 1]
}

fn main() {
    let baz = [[0_u8; 1]; 8];

    let _: [u8; 4] = foo(foo(foo(baz[0], baz[1]), foo(baz[2], baz[3])), foo(foo(baz[4], baz[5]), foo(baz[6], baz[7])));
    //let _: [u8; 3] = foo(foo(baz[0], baz[1]), foo(baz[2], baz[3]));
}
