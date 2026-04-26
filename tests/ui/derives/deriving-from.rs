//@ edition: 2021
//@ run-pass

#![feature(derive_from)]

use core::from::From;

#[derive(From)]
struct TupleSimple(u32);

#[derive(From)]
struct TupleNonPathType([u32; 4]);

#[derive(From)]
struct TupleWithRef<'a, T>(&'a T);

#[derive(From)]
struct TupleSWithBound<T: std::fmt::Debug>(T);

#[derive(From)]
struct RawIdentifier {
    r#use: u32,
}

#[derive(From)]
struct Field {
    foo: bool,
}

#[derive(From)]
struct Const<const C: usize> {
    foo: [u32; C],
}

fn main() {
    let a = 42u32;
    let b: [u32; 4] = [0, 1, 2, 3];
    let c = true;

    let s1: TupleSimple = a.into();
    assert_eq!(s1.0, a);

    let s2: TupleNonPathType = b.into();
    assert_eq!(s2.0, b);

    let s3: TupleWithRef<u32> = (&a).into();
    assert_eq!(s3.0, &a);

    let s4: TupleSWithBound<u32> = a.into();
    assert_eq!(s4.0, a);

    let s5: RawIdentifier = a.into();
    assert_eq!(s5.r#use, a);

    let s6: Field = c.into();
    assert_eq!(s6.foo, c);

    let s7: Const<4> = b.into();
    assert_eq!(s7.foo, b);
}
