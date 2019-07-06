// build-pass (FIXME(62277): could be check-pass?)
#![feature(const_fn_union)]

union U {
    a: usize,
    b: usize,
}

const C: U = U { a: 10 };

fn main() {
    let a: [u8; unsafe { C.a }];
    let b: [u8; unsafe { C.b }];
}
