#![feature(type_alias_impl_trait)]
#![warn(clippy::new_ret_no_self)]
#![allow(dead_code)]

fn main() {}

trait R {
    type Item;
}

trait Q {
    type Item;
    type Item2;
}

struct S;

impl R for S {
    type Item = Self;
}

impl S {
    // should not trigger the lint
    pub fn new() -> impl R<Item = Self> {
        S
    }
}

struct S2;

impl R for S2 {
    type Item = Self;
}

impl S2 {
    // should not trigger the lint
    pub fn new(_: String) -> impl R<Item = Self> {
        S2
    }
}

struct S3;

impl R for S3 {
    type Item = u32;
}

impl S3 {
    // should trigger the lint
    pub fn new(_: String) -> impl R<Item = u32> {
        S3
    }
}

struct S4;

impl Q for S4 {
    type Item = u32;
    type Item2 = Self;
}

impl S4 {
    // should not trigger the lint
    pub fn new(_: String) -> impl Q<Item = u32, Item2 = Self> {
        S4
    }
}

struct T;

impl T {
    // should not trigger lint
    pub fn new() -> Self {
        unimplemented!();
    }
}

struct U;

impl U {
    // should trigger lint
    pub fn new() -> u32 {
        unimplemented!();
    }
}

struct V;

impl V {
    // should trigger lint
    pub fn new(_: String) -> u32 {
        unimplemented!();
    }
}

struct TupleReturnerOk;

impl TupleReturnerOk {
    // should not trigger lint
    pub fn new() -> (Self, u32) {
        unimplemented!();
    }
}

struct TupleReturnerOk2;

impl TupleReturnerOk2 {
    // should not trigger lint (it doesn't matter which element in the tuple is Self)
    pub fn new() -> (u32, Self) {
        unimplemented!();
    }
}

struct TupleReturnerOk3;

impl TupleReturnerOk3 {
    // should not trigger lint (tuple can contain multiple Self)
    pub fn new() -> (Self, Self) {
        unimplemented!();
    }
}

struct TupleReturnerBad;

impl TupleReturnerBad {
    // should trigger lint
    pub fn new() -> (u32, u32) {
        unimplemented!();
    }
}

struct MutPointerReturnerOk;

impl MutPointerReturnerOk {
    // should not trigger lint
    pub fn new() -> *mut Self {
        unimplemented!();
    }
}

struct ConstPointerReturnerOk2;

impl ConstPointerReturnerOk2 {
    // should not trigger lint
    pub fn new() -> *const Self {
        unimplemented!();
    }
}

struct MutPointerReturnerBad;

impl MutPointerReturnerBad {
    // should trigger lint
    pub fn new() -> *mut V {
        unimplemented!();
    }
}

struct GenericReturnerOk;

impl GenericReturnerOk {
    // should not trigger lint
    pub fn new() -> Option<Self> {
        unimplemented!();
    }
}

struct GenericReturnerBad;

impl GenericReturnerBad {
    // should trigger lint
    pub fn new() -> Option<u32> {
        unimplemented!();
    }
}

struct NestedReturnerOk;

impl NestedReturnerOk {
    // should not trigger lint
    pub fn new() -> (Option<Self>, u32) {
        unimplemented!();
    }
}

struct NestedReturnerOk2;

impl NestedReturnerOk2 {
    // should not trigger lint
    pub fn new() -> ((Self, u32), u32) {
        unimplemented!();
    }
}

struct NestedReturnerOk3;

impl NestedReturnerOk3 {
    // should not trigger lint
    pub fn new() -> Option<(Self, u32)> {
        unimplemented!();
    }
}

struct WithLifetime<'a> {
    cat: &'a str,
}

impl<'a> WithLifetime<'a> {
    // should not trigger the lint, because the lifetimes are different
    pub fn new<'b: 'a>(s: &'b str) -> WithLifetime<'b> {
        unimplemented!();
    }
}

mod issue5435 {
    struct V;

    pub trait TraitRetSelf {
        // should not trigger lint
        fn new() -> Self;
    }

    pub trait TraitRet {
        // should trigger lint as we are in trait definition
        fn new() -> String;
    }
    pub struct StructRet;
    impl TraitRet for StructRet {
        // should not trigger lint as we are in the impl block
        fn new() -> String {
            unimplemented!();
        }
    }

    pub trait TraitRet2 {
        // should trigger lint
        fn new(_: String) -> String;
    }

    trait TupleReturnerOk {
        // should not trigger lint
        fn new() -> (Self, u32)
        where
            Self: Sized,
        {
            unimplemented!();
        }
    }

    trait TupleReturnerOk2 {
        // should not trigger lint (it doesn't matter which element in the tuple is Self)
        fn new() -> (u32, Self)
        where
            Self: Sized,
        {
            unimplemented!();
        }
    }

    trait TupleReturnerOk3 {
        // should not trigger lint (tuple can contain multiple Self)
        fn new() -> (Self, Self)
        where
            Self: Sized,
        {
            unimplemented!();
        }
    }

    trait TupleReturnerBad {
        // should trigger lint
        fn new() -> (u32, u32) {
            unimplemented!();
        }
    }

    trait MutPointerReturnerOk {
        // should not trigger lint
        fn new() -> *mut Self
        where
            Self: Sized,
        {
            unimplemented!();
        }
    }

    trait ConstPointerReturnerOk2 {
        // should not trigger lint
        fn new() -> *const Self
        where
            Self: Sized,
        {
            unimplemented!();
        }
    }

    trait MutPointerReturnerBad {
        // should trigger lint
        fn new() -> *mut V {
            unimplemented!();
        }
    }

    trait GenericReturnerOk {
        // should not trigger lint
        fn new() -> Option<Self>
        where
            Self: Sized,
        {
            unimplemented!();
        }
    }

    trait NestedReturnerOk {
        // should not trigger lint
        fn new() -> (Option<Self>, u32)
        where
            Self: Sized,
        {
            unimplemented!();
        }
    }

    trait NestedReturnerOk2 {
        // should not trigger lint
        fn new() -> ((Self, u32), u32)
        where
            Self: Sized,
        {
            unimplemented!();
        }
    }

    trait NestedReturnerOk3 {
        // should not trigger lint
        fn new() -> Option<(Self, u32)>
        where
            Self: Sized,
        {
            unimplemented!();
        }
    }
}

// issue #1724
struct RetOtherSelf<T>(T);
struct RetOtherSelfWrapper<T>(T);

impl RetOtherSelf<T> {
    fn new(t: T) -> RetOtherSelf<RetOtherSelfWrapper<T>> {
        RetOtherSelf(RetOtherSelfWrapper(t))
    }
}

mod issue7344 {
    struct RetImplTraitSelf<T>(T);

    impl<T> RetImplTraitSelf<T> {
        // should not trigger lint
        fn new(t: T) -> impl Into<Self> {
            Self(t)
        }
    }

    struct RetImplTraitNoSelf<T>(T);

    impl<T> RetImplTraitNoSelf<T> {
        // should trigger lint
        fn new(t: T) -> impl Into<i32> {
            1
        }
    }

    trait Trait2<T, U> {}
    impl<T, U> Trait2<T, U> for () {}

    struct RetImplTraitSelf2<T>(T);

    impl<T> RetImplTraitSelf2<T> {
        // should not trigger lint
        fn new(t: T) -> impl Trait2<(), Self> {
            unimplemented!()
        }
    }

    struct RetImplTraitNoSelf2<T>(T);

    impl<T> RetImplTraitNoSelf2<T> {
        // should trigger lint
        fn new(t: T) -> impl Trait2<(), i32> {
            unimplemented!()
        }
    }

    struct RetImplTraitSelfAdt<'a>(&'a str);

    impl<'a> RetImplTraitSelfAdt<'a> {
        // should not trigger lint
        fn new<'b: 'a>(s: &'b str) -> impl Into<RetImplTraitSelfAdt<'b>> {
            RetImplTraitSelfAdt(s)
        }
    }
}
