//@ check-pass
//@ compile-flags: -Znext-solver

// A regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/257.
// This used to make an ambiguous error while wf-checking the global where bound on `fn foo`
// because we used to add every outlives bounds from the supertraits when adding trait goals, and
// the one on the associated type resulted in adding another normalization goal, which effectively
// made a cycle.

#![feature(rustc_attrs)]
#![expect(internal_features)]
#![rustc_no_implicit_bounds]

pub trait Bound {}
impl Bound for u8 {}

pub trait Proj {
    type Assoc;
}
impl<U: Bound> Proj for U {
    type Assoc = U;
}
impl Proj for MyField {
    type Assoc = u8;
}

pub trait Field: Proj<Assoc: Bound + 'static> {}

struct MyField;
impl Field for MyField {}

trait IdReqField {
    type This;
}
impl<F: Field> IdReqField for F {
    type This = F;
}

fn foo()
where
    <MyField as IdReqField>::This: Field,
{
}

fn main() {}
