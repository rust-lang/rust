#![allow(dead_code)]
#![feature(rustc_attrs)]

use std::cell::Cell;

// Check that a type parameter which is only used in a trait bound is
// not considered bivariant.

#[rustc_dump_variances]
struct InvariantMut<'a,A:'a,B:'a> { //~ ERROR ['a: +, A: o, B: o]
    t: &'a mut (A,B)
}

#[rustc_dump_variances]
struct InvariantCell<A> { //~ ERROR [A: o]
    t: Cell<A>
}

#[rustc_dump_variances]
struct InvariantIndirect<A> { //~ ERROR [A: o]
    t: InvariantCell<A>
}

#[rustc_dump_variances]
struct Covariant<A> { //~ ERROR [A: +]
    t: A, u: fn() -> A
}

#[rustc_dump_variances]
struct Contravariant<A> { //~ ERROR [A: -]
    t: fn(A)
}

#[rustc_dump_variances]
enum Enum<A,B,C> { //~ ERROR [A: +, B: -, C: o]
    Foo(Covariant<A>),
    Bar(Contravariant<B>),
    Zed(Covariant<C>,Contravariant<C>)
}

pub fn main() { }
