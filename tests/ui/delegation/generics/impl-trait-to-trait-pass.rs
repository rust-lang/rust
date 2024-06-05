//@ run-pass
#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait GenericTrait<T> {
    fn bar<U>(&self, x: T, y: U) -> (U, T) { (y, x) }
}
trait Trait {
    fn foo1(&self) -> i32 { 1 }
    fn foo2<T>(&self, x: T) -> T { x }
}

#[derive(Debug, PartialEq)]
struct F;
impl<T> GenericTrait<T> for F {}
impl Trait for F {}

struct S(F);
struct GenericTy<T>(F, T);

mod generic_trait {
    use super::*;

    impl<A> GenericTrait<A> for S {
        reuse GenericTrait::bar { &self.0 }
    }

    pub fn check() {
        let s = S(F);
        assert_eq!(s.bar(F, 1).1, F);
    }
}

mod generic_trait_and_type {
    use super::*;

    impl<A, B> GenericTrait<B> for GenericTy<A> {
        reuse GenericTrait::bar { &self.0 }
    }

    pub fn check() {
        let ty = GenericTy(F, F);
        assert_eq!(ty.bar("str", 1u8).1, "str");
    }
}

mod generic_type {
    use super::*;

    impl<T> Trait for GenericTy<T> {
        reuse Trait::{foo1, foo2} { &self.0 }
    }

    pub fn check() {
        let ty = GenericTy(F, "str");
        assert_eq!(ty.foo1(), 1);
        assert_eq!(ty.foo2(F), F);
    }
}

mod lifetimes {
    use std::marker::PhantomData;

    struct Struct<'a, 'b, T>(PhantomData<&'a T>, &'b u8);
    trait Trait<'a> {
        fn foo(&'a self) -> u8 { 2 }
    }

    impl<'a> Trait<'a> for u8 {}

    impl<'a, 'b, 'c, T> Trait<'c> for Struct<'b, 'a, T> {
        reuse Trait::foo { self.1 }
    }

    pub fn check() {
        let s = Struct::<u16>(PhantomData, &1);
        assert_eq!(s.foo(), 2);
    }
}

fn main() {
    generic_trait::check();
    generic_trait_and_type::check();
    generic_type::check();
    lifetimes::check();
}
