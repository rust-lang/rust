//@ check-pass

#![feature(never_type)]
#![deny(unused_unconstructable_pub_structs)]

pub struct TupleNever(!);

pub struct TupleUnit(());

pub struct TuplePhantom<T>(std::marker::PhantomData<T>);

pub struct NamedNever<T> {
    _never: !,
    _value: T,
}

pub struct NamedUnit<T> {
    _unit: (),
    _value: T,
}

pub struct NamedPhantom<T> {
    _marker: std::marker::PhantomData<T>,
}

fn main() {}
