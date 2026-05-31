//@ check-pass

#![feature(never_type)]
#![deny(unused_unconstructable_pub_structs)]

pub struct TupleNever(!);

enum Void {}

pub struct TupleUninhabited(Void);

struct Token;

pub struct TupleZst(Token);

pub struct TupleUnit(());

pub struct TuplePhantom<T>(std::marker::PhantomData<T>);

pub struct TupleNestedPhantom<T>(TuplePhantom<T>);

pub struct NamedNever<T> {
    _never: !,
    _value: T,
}

pub struct NamedUninhabited<T> {
    _void: Void,
    _value: T,
}

pub struct NamedZst<T> {
    _marker: std::marker::PhantomData<T>,
    _token: Token,
    _unit: (),
}

pub struct NamedUnit {
    _unit: (),
}

pub struct MixedUnit((), i32);

pub struct NamedPhantom<T> {
    _marker: std::marker::PhantomData<T>,
}

fn main() {}
