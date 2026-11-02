//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] build-pass
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] build-fail
//@[old] known-bug: #129372
//@ compile-flags: -Cdebuginfo=2 -Copt-level=0
//@ ignore-backends: gcc

// Regression test for #129372. With the old solver normalizing
// the output of `v.method()` ended up not fully normalizing some
// higher-ranked associated type. This then resulted in some mismatch
// later on. Fixed by the new trait solver.

pub struct Wrapper<T>(T);
struct Struct;

pub trait TraitA {
    type AssocA<'t>;
}
pub trait TraitB {
    type AssocB;
}

pub fn helper(v: impl MethodTrait) {
    let _local_that_causes_ice = v.method();
}

pub fn main() {
    helper(Wrapper(Struct));
}

pub trait MethodTrait {
    type Assoc<'a>;

    fn method(self) -> impl for<'a> FnMut(&'a ()) -> Self::Assoc<'a>;
}

impl<T: TraitB> MethodTrait for T
where
    <T as TraitB>::AssocB: TraitA,
{
    type Assoc<'a> = <T::AssocB as TraitA>::AssocA<'a>;

    fn method(self) -> impl for<'a> FnMut(&'a ()) -> Self::Assoc<'a> {
        move |_| loop {}
    }
}

impl<T, B> TraitB for Wrapper<B>
where
    B: TraitB<AssocB = T>,
{
    type AssocB = T;
}

impl TraitB for Struct {
    type AssocB = Struct;
}

impl TraitA for Struct {
    type AssocA<'t> = Self;
}
