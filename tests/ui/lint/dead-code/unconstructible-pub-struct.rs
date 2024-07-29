#![feature(never_type)]
#![deny(dead_code)]

pub struct T1(!);
pub struct T2(());
pub struct T3<X>(std::marker::PhantomData<X>);

pub struct T4 {
    _x: !,
}

pub struct T5<X> {
    _x: !,
    _y: X,
}

pub struct T6 {
    _x: (),
}

pub struct T7<X> {
    _x: (),
    _y: X,
}

pub struct T8<X> {
    _x: std::marker::PhantomData<X>,
}

pub struct T9<X> { //~ ERROR struct `T9` is never constructed
    _x: std::marker::PhantomData<X>,
    _y: i32,
}

fn main() {}
