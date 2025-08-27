#![feature(never_type)]
#![deny(unconstructable_pub_struct)]

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

pub struct T9<X> { //~ ERROR: pub struct `T9` is unconstructable externally and never constructed locally
    _x: std::marker::PhantomData<X>,
    _y: i32,
}

pub struct _T10(i32);

mod pri {
    pub struct Unreachable(i32);
}

pub struct NeverConstructed(i32); //~ ERROR: pub struct `NeverConstructed` is unconstructable externally and never constructed locally

impl NeverConstructed {
    pub fn not_construct_self(&self) {}
}

impl Clone for NeverConstructed {
    fn clone(&self) -> NeverConstructed {
        NeverConstructed(0)
    }
}

pub trait Trait {
    fn not_construct_self(&self);
}

impl Trait for NeverConstructed {
    fn not_construct_self(&self) {
        self.0;
    }
}

pub struct Constructed(i32);

impl Constructed {
    pub fn construct_self() -> Self {
        Constructed(0)
    }
}

impl Clone for Constructed {
    fn clone(&self) -> Constructed {
        Constructed(0)
    }
}

impl Trait for Constructed {
    fn not_construct_self(&self) {
        self.0;
    }
}

fn main() {}
