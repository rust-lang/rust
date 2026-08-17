//@ build-pass
//@ compile-flags: -Znext-solver

//! Consider:
//!
//! goal is <StructTailHasAnonConst as Pointee>::Metadata == ?0, in a typing context of
//! ErasedNotCoherence
//!
//! `consider_builtin_pointee_candidate` looks at StructTailHasAnonConst, realizes it's an ADT,
//! fetches the struct tail (which is `S<{ 2 + 2 }>`), and calls `instantiate_normalizes_to_term`
//! with the result of `<Struct<{ 2 + 2 }> as Pointee>::Metadata`
//!
//! `instantiate_normalizes_to_term` `.eq()`s `<Struct<{ 2 + 2 }> as Pointee>::Metadata` and `?0`
//!
//! this eagerly normalizes, which normalizes the anon const, which fails due to ErasedNotCoherence
//!
//! this causes the `.eq()` in `instantiate_normalizes_to_term` to fail, which used to have an
//! unwrap, which ICEd

#![feature(ptr_metadata)]

struct S<const N: usize>;

struct StructTailHasAnonConst(S<{ 2 + 2 }>);

fn main() {
    let y: <StructTailHasAnonConst as std::ptr::Pointee>::Metadata;
}
