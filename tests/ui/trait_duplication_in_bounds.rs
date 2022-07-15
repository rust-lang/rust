#![deny(clippy::trait_duplication_in_bounds)]
#![allow(unused)]

use std::collections::BTreeMap;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

fn bad_foo<T: Clone + Default, Z: Copy>(arg0: T, arg1: Z)
where
    T: Clone,
    T: Default,
{
    unimplemented!();
}

fn good_bar<T: Clone + Default>(arg: T) {
    unimplemented!();
}

fn good_foo<T>(arg: T)
where
    T: Clone + Default,
{
    unimplemented!();
}

fn good_foobar<T: Default>(arg: T)
where
    T: Clone,
{
    unimplemented!();
}

trait T: Default {
    fn f()
    where
        Self: Default;
}

trait U: Default {
    fn f()
    where
        Self: Clone;
}

trait ZZ: Default {
    fn g();
    fn h();
    fn f()
    where
        Self: Default + Clone;
}

trait BadTrait: Default + Clone {
    fn f()
    where
        Self: Default + Clone;
    fn g()
    where
        Self: Default;
    fn h()
    where
        Self: Copy;
}

#[derive(Default, Clone)]
struct Life;

impl T for Life {
    // this should not warn
    fn f() {}
}

impl U for Life {
    // this should not warn
    fn f() {}
}

// should not warn
trait Iter: Iterator {
    fn into_group_btreemap<K, V>(self) -> BTreeMap<K, Vec<V>>
    where
        Self: Iterator<Item = (K, V)> + Sized,
        K: Ord + Eq,
    {
        unimplemented!();
    }
}

struct Foo;

trait FooIter: Iterator<Item = Foo> {
    fn bar()
    where
        Self: Iterator<Item = Foo>,
    {
    }
}

// This should not lint
fn impl_trait(_: impl AsRef<str>, _: impl AsRef<str>) {}

mod repeated_where_clauses_or_trait_bounds {
    fn bad_foo<T: Clone + Clone + Clone + Copy, U: Clone + Copy>(arg0: T, argo1: U) {
        unimplemented!();
    }

    fn bad_bar<T, U>(arg0: T, arg1: U)
    where
        T: Clone + Clone + Clone + Copy,
        U: Clone + Copy,
    {
        unimplemented!();
    }

    fn good_bar<T: Clone + Copy, U: Clone + Copy>(arg0: T, arg1: U) {
        unimplemented!();
    }

    fn good_foo<T, U>(arg0: T, arg1: U)
    where
        T: Clone + Copy,
        U: Clone + Copy,
    {
        unimplemented!();
    }

    trait GoodSelfTraitBound: Clone + Copy {
        fn f();
    }

    trait GoodSelfWhereClause {
        fn f()
        where
            Self: Clone + Copy;
    }

    trait BadSelfTraitBound: Clone + Clone + Clone {
        fn f();
    }

    trait BadSelfWhereClause {
        fn f()
        where
            Self: Clone + Clone + Clone;
    }

    trait GoodTraitBound<T: Clone + Copy, U: Clone + Copy> {
        fn f();
    }

    trait GoodWhereClause<T, U> {
        fn f()
        where
            T: Clone + Copy,
            U: Clone + Copy;
    }

    trait BadTraitBound<T: Clone + Clone + Clone + Copy, U: Clone + Copy> {
        fn f();
    }

    trait BadWhereClause<T, U> {
        fn f()
        where
            T: Clone + Clone + Clone + Copy,
            U: Clone + Copy;
    }

    struct GoodStructBound<T: Clone + Copy, U: Clone + Copy> {
        t: T,
        u: U,
    }

    impl<T: Clone + Copy, U: Clone + Copy> GoodTraitBound<T, U> for GoodStructBound<T, U> {
        // this should not warn
        fn f() {}
    }

    struct GoodStructWhereClause;

    impl<T, U> GoodTraitBound<T, U> for GoodStructWhereClause
    where
        T: Clone + Copy,
        U: Clone + Copy,
    {
        // this should not warn
        fn f() {}
    }

    fn no_error_separate_arg_bounds(program: impl AsRef<()>, dir: impl AsRef<()>, args: &[impl AsRef<()>]) {}

    trait GenericTrait<T> {}

    // This should not warn but currently does see #8757
    fn good_generic<T: GenericTrait<u64> + GenericTrait<u32>>(arg0: T) {
        unimplemented!();
    }

    fn bad_generic<T: GenericTrait<u64> + GenericTrait<u32> + GenericTrait<u64>>(arg0: T) {
        unimplemented!();
    }

    mod foo {
        pub trait Clone {}
    }

    fn qualified_path<T: std::clone::Clone + Clone + foo::Clone>(arg0: T) {
        unimplemented!();
    }
}

fn main() {}
