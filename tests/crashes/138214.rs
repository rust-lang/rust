//@ known-bug: #138214

struct Demo<X, Y: A = X, Z = <<Y as A>::U as B>::V>(X, Y, Z);

impl<V> Demo<V> {
    fn new() {}
}

pub trait A<Group = ()> {
    type U: B;
}

pub trait B {
    type V;
}
