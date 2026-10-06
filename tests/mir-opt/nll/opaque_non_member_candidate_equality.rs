//@ skip-filecheck
//@ compile-flags: -Znext-solver

#![feature(coroutines)]
#![allow(warnings)]

fn new_defining_use<F: FnOnce(T) -> R, T, R>(_: F) {}

// These defining uses share captured `'a`, but have different unrelated parent
// lifetimes. The NLL dumps check that candidate equality uses fresh local regions
// instead of constraining non-member closure/coroutine or alias arguments.

// EMIT_MIR opaque_non_member_candidate_equality.closure.nll.0.mir
fn closure<'a, 'b: 'b, 'c: 'c>(_: ()) -> impl Sized + use<'a> {
    new_defining_use(closure::<'a, 'b, 'c>);
    new_defining_use(closure::<'a, 'c, 'b>);
    || {}
}

// EMIT_MIR opaque_non_member_candidate_equality.coroutine.nll.0.mir
fn coroutine<'a, 'b: 'b, 'c: 'c>(_: ()) -> impl Sized + use<'a> {
    new_defining_use(coroutine::<'a, 'b, 'c>);
    new_defining_use(coroutine::<'a, 'c, 'b>);
    #[coroutine]
    || yield
}

// EMIT_MIR opaque_non_member_candidate_equality.nested_alias.nll.0.mir
fn nested_alias<'a, 'b: 'b, 'c: 'c>(_: ()) -> impl Sized + use<'a> {
    new_defining_use(nested_alias::<'a, 'b, 'c>);
    new_defining_use(nested_alias::<'a, 'c, 'b>);
    closure::<'a, 'b, 'c>(())
}

fn main() {}
//@ ignore-32bit
