//@ revisions: pos neg
//@[pos] check-pass

#![feature(generic_const_exprs)]
#![feature(trivial_bounds)] // only used in test case `ct_unused_1`
#![allow(incomplete_features)]

// FIXME(generic_const_exprs): Revisit this before stabilization.
// Check that we don't emit the lint `type_alias_bounds` for (eager) type aliases
// whose RHS contains a const projection (aka uneval'ed const).
// Since anon consts inherit the parent generics and predicates and we effectively
// check them before and after instantiaton for well-formedness, the type alias
// bounds are in every sense "enforced".
// Note that the test cases whose name ends in "unused" just demonstrate that this
// holds even if the const projections don't "visibly" capture any generics and/or
// predicates.
#![deny(type_alias_bounds)]

fn ct_unused_0() {
    type AliasConstUnused<T: Copy> = (T, I32<{ DATA }>);
    const DATA: i32 = 0;
    #[cfg(neg)]
    let _: AliasConstUnused<String>;
    //[neg]~^ ERROR the trait bound `String: Copy` is not satisfied
}

fn ct_unused_1() {
    #[allow(trivial_bounds)]
    type AliasConstUnused where String: Copy = I32<{ 0; 0 }>;
    //[neg]~^ ERROR entering unreachable code
    #[cfg(neg)]
    let _: AliasConstUnused;
    //[neg]~^ ERROR the trait bound `String: Copy` is not satisfied
}

fn fn_unused() {
    type AliasFnUnused<T: Copy> = (T, I32<{ code() }>);
    const fn code() -> i32 { 0 }
    #[cfg(neg)]
    let _: AliasFnUnused<String>;
    //[neg]~^ ERROR the trait bound `String: Copy` is not satisfied
}

trait Trait {
    type Proj;
    const DATA: i32;
}

impl Trait for String {
    type Proj = i32;
    const DATA: i32 = 0;
}

// Regression test for issue #94398.
fn assoc_ct_used() {
    type AliasAssocConstUsed<T: Trait + Copy> = I32<{ T::DATA }>;
    #[cfg(neg)]
    let _: AliasAssocConstUsed<String>;
    //[neg]~^ ERROR the trait bound `String: Copy` is not satisfied
}

fn fn_used() {
    type AliasFnUsed<T: Trait + Copy> = I32<{ code::<T>() }>;
    const fn code<T: Trait>() -> i32 { T::DATA }
    #[cfg(neg)]
    let _: AliasFnUsed<String>;
    //[neg]~^ ERROR the trait bound `String: Copy` is not satisfied
}

struct I32<const N: i32>;

fn main() {}
