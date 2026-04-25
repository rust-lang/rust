//! This test should be part of path-to-non-type-const.rs, and should pass. However, we are holding
//! off on implementing paths to IACs until a refactoring of how IAC generics are represented.
//@ revisions: old next
//@[next] compile-flags: -Znext-solver

#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

struct StructImpl;
struct GenericStructImpl<const A: usize>;

impl StructImpl {
    const INHERENT: usize = 1;
}

impl<const A: usize> GenericStructImpl<A> {
    const INHERENT: usize = A;
}

struct Struct<const N: usize>;

fn main() {
    let _ = Struct::<{ StructImpl::INHERENT }>;
    //~^ ERROR use of `const` in the type system not defined as `type const`
    let _ = Struct::<{ GenericStructImpl::<2>::INHERENT }>;
    //~^ ERROR use of `const` in the type system not defined as `type const`
}
