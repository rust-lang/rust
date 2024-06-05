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

    impl S {
        reuse GenericTrait::bar { &self.0 }
    }

    pub fn check() {
        let s = S(F);
        assert_eq!(s.bar(F, 1).1, F);
        assert_eq!(<S as GenericTrait<_>>::bar(&s, F, 1).1, F);
    }
}

mod generic_trait_and_type {
    use super::*;

    impl<A, B> GenericTrait<B> for GenericTy<A> {
        reuse GenericTrait::bar { &self.0 }
    }

    impl<A> GenericTy<A> {
        reuse GenericTrait::bar { &self.0 }
    }

    pub fn check() {
        let ty = GenericTy(F, F);
        assert_eq!(ty.bar("str", 1u8).1, "str");
        assert_eq!(<GenericTy<F> as GenericTrait<_>>::bar(&ty, "str", 1u8).1, "str");
    }
}

mod generic_type {
    use super::*;

    impl<T> Trait for GenericTy<T> {
        reuse Trait::{foo1, foo2} { &self.0 }
    }

    impl<T> GenericTy<T> {
        reuse Trait::{foo1, foo2} { &self.0 }
    }

    pub fn check() {
        let ty = GenericTy(F, "str");
        assert_eq!(ty.foo1(), 1);
        assert_eq!(ty.foo2(F), F);

        assert_eq!(<GenericTy<_> as Trait>::foo1(&ty), 1);
        assert_eq!(<GenericTy<_> as Trait>::foo2(&ty, 1u16), 1u16);
    }
}

mod lifetimes {
    struct Struct<'a>(&'a u8);
    trait Trait<'a> {
        fn foo<'b>(&'a self, x: &'b u8) -> &'b u8 { x }
    }

    impl<'a> Trait<'a> for u8 {}

    impl<'a> Trait<'a> for Struct<'a> {
        reuse Trait::foo { self.0 }
    }

    impl<'a> Struct<'a> {
        reuse Trait::foo { self.0 }
    }

    pub fn check() {
        let s = Struct(&1);
        let val = &2;
        assert_eq!(*s.foo(val), *val);
        assert_eq!(*<Struct as Trait>::foo(&s, val), *val);
    }
}

fn main() {
    generic_trait::check();
    generic_trait_and_type::check();
    generic_type::check();
    lifetimes::check();
}
