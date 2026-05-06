//@ check-pass
//@ compile-flags: -Znext-solver

// Regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/257.

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

// While wf-checking the global bounds of `fn foo`, elaborating this outlives predicate triggered a
// cycle in the search graph along a particular probe path, which was not an actual solution.
// That cycle then resulted in a forced false-positive ambiguity due to a performance hack in the
// search graph and then ended up floundering the root goal evaluation.
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
