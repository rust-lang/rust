//@ edition:2024
//@ compile-flags: -Zmir-enable-passes=+Inline,+ReferencePropagation -Zlint-mir
//@ check-pass

#![feature(gen_blocks)]
#![allow(unused_variables, path_statements)]

struct NonCopy(i32);

gen fn refs<'a, 'b>(x: &'a i32, y: &'b i32, z: &'b i32) -> &'b i32 {
    yield y;
    z;
}

gen fn moves(x: NonCopy, y: NonCopy, z: NonCopy) -> NonCopy {
    yield y;
    z;
}

fn main() {
    let z = 3;
    let mut refs_iter = refs(&1, &2, &z);
    assert_eq!(refs_iter.next(), Some(&3));

    let mut moves_iter = moves(NonCopy(1), NonCopy(2), NonCopy(3));
    assert!(matches!(moves_iter.next(), Some(_)));
}
