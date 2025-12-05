//@ revisions: a b

#![crate_type = "lib"]
#![feature(alloc_error_handler)]
#![feature(cfg_accessible)]
#![feature(cfg_eval)]
#![feature(custom_test_frameworks)]
#![feature(derive_const)]
#![feature(where_clause_attrs)]
#![allow(soft_unstable)]

use std::marker::PhantomData;

#[cfg(a)]
trait TraitA {}

#[cfg(b)]
trait TraitB {}

#[cfg_attr(a, cfg(a))]
trait TraitAA {}

#[cfg_attr(b, cfg(b))]
trait TraitBB {}

trait A<T>
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
    #[cfg_attr(a, cfg(a))] T: TraitAA,
    #[cfg_attr(b, cfg(b))] T: TraitBB,
    #[derive(Clone)] ():,
    //~^ ERROR most attributes are not supported in `where` clauses
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[rustfmt::skip] ():, //~ ERROR most attributes are not supported in `where` clauses
{
    type B<U>
    where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB,
        #[cfg_attr(a, cfg(a))] U: TraitAA,
        #[cfg_attr(b, cfg(b))] U: TraitBB,
        #[derive(Clone)] ():,
        //~^ ERROR most attributes are not supported in `where` clauses
        //~| ERROR expected non-macro attribute, found attribute macro `derive`
        #[rustfmt::skip] ():; //~ ERROR most attributes are not supported in `where` clauses

    fn foo<U>(&self)
    where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB,
        #[cfg_attr(a, cfg(a))] U: TraitAA,
        #[cfg_attr(b, cfg(b))] U: TraitBB,
        #[derive(Clone)] ():,
        //~^ ERROR most attributes are not supported in `where` clauses
        //~| ERROR expected non-macro attribute, found attribute macro `derive`
        #[rustfmt::skip] ():; //~ ERROR most attributes are not supported in `where` clauses
}

impl<T> A<T> for T
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
    #[cfg_attr(a, cfg(a))] T: TraitAA,
    #[cfg_attr(b, cfg(b))] T: TraitBB,
    #[derive(Clone)] ():,
    //~^ ERROR most attributes are not supported in `where` clauses
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[rustfmt::skip] ():, //~ ERROR most attributes are not supported in `where` clauses
{
    type B<U> = () where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB,
        #[cfg_attr(a, cfg(a))] U: TraitAA,
        #[cfg_attr(b, cfg(b))] U: TraitBB,
        #[derive(Clone)] ():,
        //~^ ERROR most attributes are not supported in `where` clauses
        //~| ERROR expected non-macro attribute, found attribute macro `derive`
        #[rustfmt::skip] ():; //~ ERROR most attributes are not supported in `where` clauses

    fn foo<U>(&self)
    where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB,
        #[cfg_attr(a, cfg(a))] U: TraitAA,
        #[cfg_attr(b, cfg(b))] U: TraitBB,
        #[derive(Clone)] ():,
        //~^ ERROR most attributes are not supported in `where` clauses
        //~| ERROR expected non-macro attribute, found attribute macro `derive`
        #[rustfmt::skip] ():, //~ ERROR most attributes are not supported in `where` clauses
    {}
}

struct C<T>
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
    #[cfg_attr(a, cfg(a))] T: TraitAA,
    #[cfg_attr(b, cfg(b))] T: TraitBB,
    #[derive(Clone)] ():,
    //~^ ERROR most attributes are not supported in `where` clauses
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[rustfmt::skip] ():, //~ ERROR most attributes are not supported in `where` clauses
{
    _t: PhantomData<T>,
}

union D<T>
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
    #[cfg_attr(a, cfg(a))] T: TraitAA,
    #[cfg_attr(b, cfg(b))] T: TraitBB,
    #[derive(Clone)] ():,
    //~^ ERROR most attributes are not supported in `where` clauses
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[rustfmt::skip] ():, //~ ERROR most attributes are not supported in `where` clauses
{

    _t: PhantomData<T>,
}

enum E<T>
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
    #[cfg_attr(a, cfg(a))] T: TraitAA,
    #[cfg_attr(b, cfg(b))] T: TraitBB,
    #[derive(Clone)] ():,
    //~^ ERROR most attributes are not supported in `where` clauses
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[rustfmt::skip] ():, //~ ERROR most attributes are not supported in `where` clauses
{
    E(PhantomData<T>),
}

#[allow(type_alias_bounds)]
type F<T>
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
    #[cfg_attr(a, cfg(a))] T: TraitAA,
    #[cfg_attr(b, cfg(b))] T: TraitBB,
    #[derive(Clone)] ():,
    //~^ ERROR most attributes are not supported in `where` clauses
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[rustfmt::skip] ():, //~ ERROR most attributes are not supported in `where` clauses
= T;

impl<T> C<T> where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
    #[cfg_attr(a, cfg(a))] T: TraitAA,
    #[cfg_attr(b, cfg(b))] T: TraitBB,
    #[derive(Clone)] ():,
    //~^ ERROR most attributes are not supported in `where` clauses
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[rustfmt::skip] ():, //~ ERROR most attributes are not supported in `where` clauses
{
    fn new<U>() where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB,
        #[cfg_attr(a, cfg(a))] U: TraitAA,
        #[cfg_attr(b, cfg(b))] U: TraitBB,
        #[derive(Clone)] ():,
        //~^ ERROR most attributes are not supported in `where` clauses
        //~| ERROR expected non-macro attribute, found attribute macro `derive`
        #[rustfmt::skip] ():, //~ ERROR most attributes are not supported in `where` clauses
    {}
}

fn foo<T>()
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
    #[cfg_attr(a, cfg(a))] T: TraitAA,
    #[cfg_attr(b, cfg(b))] T: TraitBB,
    #[derive(Clone)] ():,
    //~^ ERROR most attributes are not supported in `where` clauses
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[rustfmt::skip] ():, //~ ERROR most attributes are not supported in `where` clauses
{
}
