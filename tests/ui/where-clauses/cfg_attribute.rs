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
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[global_allocator] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
    #[test] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test`
    #[alloc_error_handler] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
    #[bench] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `bench`
    #[cfg_accessible] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
    #[cfg_eval] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
    #[derive_const(Clone)] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
    #[test_case] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test_case`
    #[rustfmt::skip] ():, //~ ERROR most attributes in `where` clauses are not supported
{
    type B<U>
    where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB,
        #[cfg_attr(a, cfg(a))] U: TraitAA,
        #[cfg_attr(b, cfg(b))] U: TraitBB,
        #[derive(Clone)] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `derive`
        #[global_allocator] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
        #[test] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `test`
        #[alloc_error_handler] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
        #[bench] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `bench`
        #[cfg_accessible] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
        #[cfg_eval] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
        #[derive_const(Clone)] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
        #[test_case] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `test_case`
        #[rustfmt::skip] ():; //~ ERROR most attributes in `where` clauses are not supported

    fn foo<U>(&self)
    where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB,
        #[cfg_attr(a, cfg(a))] U: TraitAA,
        #[cfg_attr(b, cfg(b))] U: TraitBB,
        #[derive(Clone)] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `derive`
        #[global_allocator] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
        #[test] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `test`
        #[alloc_error_handler] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
        #[bench] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `bench`
        #[cfg_accessible] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
        #[cfg_eval] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
        #[derive_const(Clone)] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
        #[test_case] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `test_case`
        #[rustfmt::skip] ():; //~ ERROR most attributes in `where` clauses are not supported
}

impl<T> A<T> for T
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
    #[cfg_attr(a, cfg(a))] T: TraitAA,
    #[cfg_attr(b, cfg(b))] T: TraitBB,
    #[derive(Clone)] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[global_allocator] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
    #[test] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test`
    #[alloc_error_handler] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
    #[bench] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `bench`
    #[cfg_accessible] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
    #[cfg_eval] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
    #[derive_const(Clone)] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
    #[test_case] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test_case`
    #[rustfmt::skip] ():, //~ ERROR most attributes in `where` clauses are not supported
{
    type B<U> = () where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB,
        #[cfg_attr(a, cfg(a))] U: TraitAA,
        #[cfg_attr(b, cfg(b))] U: TraitBB,
        #[derive(Clone)] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `derive`
        #[global_allocator] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
        #[test] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `test`
        #[alloc_error_handler] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
        #[bench] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `bench`
        #[cfg_accessible] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
        #[cfg_eval] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
        #[derive_const(Clone)] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
        #[test_case] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `test_case`
        #[rustfmt::skip] ():; //~ ERROR most attributes in `where` clauses are not supported

    fn foo<U>(&self)
    where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB,
        #[cfg_attr(a, cfg(a))] U: TraitAA,
        #[cfg_attr(b, cfg(b))] U: TraitBB,
        #[derive(Clone)] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `derive`
        #[global_allocator] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
        #[test] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `test`
        #[alloc_error_handler] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
        #[bench] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `bench`
        #[cfg_accessible] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
        #[cfg_eval] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
        #[derive_const(Clone)] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
        #[test_case] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `test_case`
        #[rustfmt::skip] ():, //~ ERROR most attributes in `where` clauses are not supported
    {}
}

struct C<T>
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
    #[cfg_attr(a, cfg(a))] T: TraitAA,
    #[cfg_attr(b, cfg(b))] T: TraitBB,
    #[derive(Clone)] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[global_allocator] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
    #[test] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test`
    #[alloc_error_handler] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
    #[bench] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `bench`
    #[cfg_accessible] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
    #[cfg_eval] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
    #[derive_const(Clone)] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
    #[test_case] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test_case`
    #[rustfmt::skip] ():, //~ ERROR most attributes in `where` clauses are not supported
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
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[global_allocator] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
    #[test] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test`
    #[alloc_error_handler] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
    #[bench] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `bench`
    #[cfg_accessible] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
    #[cfg_eval] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
    #[derive_const(Clone)] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
    #[test_case] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test_case`
    #[rustfmt::skip] ():, //~ ERROR most attributes in `where` clauses are not supported
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
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[global_allocator] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
    #[test] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test`
    #[alloc_error_handler] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
    #[bench] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `bench`
    #[cfg_accessible] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
    #[cfg_eval] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
    #[derive_const(Clone)] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
    #[test_case] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test_case`
    #[rustfmt::skip] ():, //~ ERROR most attributes in `where` clauses are not supported
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
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[global_allocator] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
    #[test] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test`
    #[alloc_error_handler] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
    #[bench] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `bench`
    #[cfg_accessible] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
    #[cfg_eval] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
    #[derive_const(Clone)] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
    #[test_case] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test_case`
    #[rustfmt::skip] ():, //~ ERROR most attributes in `where` clauses are not supported
= T;

impl<T> C<T> where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
    #[cfg_attr(a, cfg(a))] T: TraitAA,
    #[cfg_attr(b, cfg(b))] T: TraitBB,
    #[derive(Clone)] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[global_allocator] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
    #[test] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test`
    #[alloc_error_handler] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
    #[bench] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `bench`
    #[cfg_accessible] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
    #[cfg_eval] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
    #[derive_const(Clone)] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
    #[test_case] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test_case`
    #[rustfmt::skip] ():, //~ ERROR most attributes in `where` clauses are not supported
{
    fn new<U>() where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB,
        #[cfg_attr(a, cfg(a))] U: TraitAA,
        #[cfg_attr(b, cfg(b))] U: TraitBB,
        #[derive(Clone)] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `derive`
        #[global_allocator] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
        #[test] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `test`
        #[alloc_error_handler] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
        #[bench] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `bench`
        #[cfg_accessible] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
        #[cfg_eval] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
        #[derive_const(Clone)] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
        #[test_case] ():,
        //~^ ERROR most attributes in `where` clauses are not supported
        //~| ERROR expected non-macro attribute, found attribute macro `test_case`
        #[rustfmt::skip] ():, //~ ERROR most attributes in `where` clauses are not supported
    {}
}

fn foo<T>()
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
    #[cfg_attr(a, cfg(a))] T: TraitAA,
    #[cfg_attr(b, cfg(b))] T: TraitBB,
    #[derive(Clone)] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[global_allocator] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
    #[test] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test`
    #[alloc_error_handler] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
    #[bench] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `bench`
    #[cfg_accessible] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
    #[cfg_eval] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
    #[derive_const(Clone)] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
    #[test_case] ():,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test_case`
    #[rustfmt::skip] ():, //~ ERROR most attributes in `where` clauses are not supported
{
}
