//@ edition: 2024
//@ check-pass
#![feature(gen_blocks)]

// make sure gen fn captures lifetimes in its signature

gen fn foo<'a, 'b>(x: &'a i32, y: &'b i32, z: &'b i32) -> &'b i32 {
    yield y;
    yield z;
}

fn main() {
    let z = 3;
    let mut iter = foo(&1, &2, &z);
    assert_eq!(iter.next(), Some(&2));
    assert_eq!(iter.next(), Some(&3));
    assert_eq!(iter.next(), None);
}
