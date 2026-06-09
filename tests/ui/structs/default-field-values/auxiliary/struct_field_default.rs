#![feature(default_field_values)]

pub struct A {
    pub a: isize = 42,
}

struct Priv;

pub struct B {
    pub a: Priv = Priv,
}

pub struct C {
    pub a: Priv,
}
