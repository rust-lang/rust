// Make sure that built-in derives don't rely on the user not declaring certain
// names to work properly.

//@ check-pass

#![allow(nonstandard_style)]
#![feature(decl_macro)]

use std::prelude::v1::test as inline;

static f: () = ();
static cmp: () = ();
static other: () = ();
static state: () = ();
static __self_0_0: () = ();
static __self_1_0: () = ();
static __self_vi: () = ();
static __arg_1_0: () = ();
static debug_trait_builder: () = ();

struct isize;
trait i16 {}

trait MethodsInDerives: Sized {
    fn debug_tuple(self) {}
    fn debug_struct(self) {}
    fn field(self) {}
    fn finish(self) {}
    fn clone(self) {}
    fn cmp(self) {}
    fn partial_cmp(self) {}
    fn eq(self) {}
    fn ne(self) {}
    fn le(self) {}
    fn lt(self) {}
    fn ge(self) {}
    fn gt(self) {}
    fn hash(self) {}
}

trait GenericAny<T, U> {}
impl<S, T, U> GenericAny<T, U> for S {}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
enum __H { V(i32), }

#[repr(i16)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
enum W { A, B }

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
struct X<A: GenericAny<A, self::X<i32>>> {
    A: A,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
struct Y<B>(B)
where
    B: From<B>;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
enum Z<C> {
    C(C),
    B { C: C },
}

// Make sure that we aren't using `self::` in paths, since it doesn't work in
// non-module scopes.
const NON_MODULE: () = {
    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
    enum __H { V(i32), }

    #[repr(i16)]
    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
    enum W { A, B }

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
    struct X<A: Fn(A) -> self::X<i32>> {
        A: A,
    }

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
    struct Y<B>(B)
    where
        B: From<B>;

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
    enum Z<C> {
        C(C),
        B { C: C },
    }
};

macro m() {
    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
    enum __H { V(i32), }

    #[repr(i16)]
    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
    enum W { A, B }

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
    struct X<A: GenericAny<A, self::X<i32>>> {
        A: A,
    }

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
    struct Y<B>(B)
    where
        B: From<B>;

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
    enum Z<C> {
        C(C),
        B { C: C },
    }
}

m!();

fn main() {}
