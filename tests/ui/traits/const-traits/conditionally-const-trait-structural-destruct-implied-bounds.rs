//@ revisions: next old
//@ [next] compile-flags: -Znext-solver

#![feature(const_trait_impl, const_destruct)]

use std::marker::Destruct;
use std::mem::ManuallyDrop;

const fn ensure_const_destruct<T: const Destruct>(_t: T) {}

const trait TraitA {
    fn f(self)
    where
        Self: [const] Destruct;
}

impl<T: [const] TraitA> const TraitA for [T; 1] {
    fn f(self)
    where
        Self: [const] Destruct,
    {
        let [t] = self;
        t.f()
    }
}

pub struct Foo<T, U> {
    pub pub_field: T,
    u: U,
}

pub struct Bar<T>(pub T);

impl<T: [const] TraitA, U: [const] TraitA> const TraitA for Foo<Bar<T>, U> {
    fn f(self)
    where
        Self: [const] Destruct,
    {
        let Foo { pub_field: Bar(t), u } = self;
        t.f();
        u.f(); //~ ERROR the trait bound `U: [const] Destruct` is not satisfied
    }
}

const fn h<T>(x: Option<T>)
where
    Option<T>: [const] Destruct,
{
    ensure_const_destruct(x.unwrap())
    //~^ ERROR the trait bound `T: const Destruct` is not satisfied
}

const trait TraitB {
    fn g(self)
    where
        Self: [const] Destruct;
}

impl<T: [const] TraitB> const TraitA for Bar<T> {
    fn f(self)
    where
        Self: [const] Destruct,
    {
        self.0.g()
        //~^ ERROR the trait bound `T: [const] Destruct` is not satisfied
    }
}

impl<T: [const] TraitA> const TraitA for ManuallyDrop<T> {
    fn f(self)
    where
        Self: [const] Destruct,
    {
        ManuallyDrop::into_inner(self).f()
        //~^ ERROR the trait bound `T: [const] Destruct` is not satisfied
    }
}

fn main() {}
