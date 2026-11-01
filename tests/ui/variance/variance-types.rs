#![feature(rustc_attrs)]
#![feature(covariant_unsafe_cell, phantom_variance_markers)]

use std::cell::Cell;
use std::marker::*;

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

#[rustc_dump_variances]
struct PhantomDataIsCovariant<A> { //~ ERROR [A: +]
    t: std::marker::PhantomData<A>
}

#[rustc_dump_variances]
struct UnsafeCellIsInvariant<A> { //~ ERROR [A: o]
    t: std::cell::UnsafeCell<A>
}

#[rustc_dump_variances]
struct CovariantUnsafeCellIsCovariant<A> { //~ ERROR [A: +]
    t: std::cell::CovariantUnsafeCell<A>
}

#[rustc_dump_variances]
struct Types<T: ?Sized, U: ?Sized, V: ?Sized>(
    PhantomCovariant<T>, PhantomContravariant<U>, PhantomInvariant<V>,
);
//~^^^ ERROR [T: +, U: -, V: o]
#[rustc_dump_variances]
struct Lifetimes<'a, 'b, 'c>(
    PhantomCovariantLifetime<'a>, PhantomContravariantLifetime<'b>, PhantomInvariantLifetime<'c>,
);
//~^^^ ERROR ['a: +, 'b: -, 'c: o]

pub fn main() { }
