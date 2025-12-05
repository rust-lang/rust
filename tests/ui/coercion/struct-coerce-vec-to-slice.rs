//! Regression test that ensures struct field literals can be coerced into slice and `Box` types

//@ check-pass

struct Thing1<'a> {
    baz: &'a [Box<isize>],
    bar: Box<u64>,
}

struct Thing2<'a> {
    baz: &'a [Box<isize>],
    bar: u64,
}

pub fn main() {
    let _a = Thing1 { baz: &[], bar: Box::new(32) };
    let _b = Thing1 { baz: &Vec::new(), bar: Box::new(32) };
    let _c = Thing2 { baz: &[], bar: 32 };
    let _d = Thing2 { baz: &Vec::new(), bar: 32 };
}
