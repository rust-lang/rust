// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -C debuginfo=full
#![crate_type = "lib"]

use std::fmt::Debug;

pub trait Cache {
    type V: Debug;

    fn store_nocache(&self);
}

pub trait Query {
    type V;
    type C: Cache<V = Self::V>;

    fn cache<T>(s: &T) -> &Self::C;
}

// EMIT_MIR dyn_trait.mk_cycle.Inline.diff
#[inline(always)]
pub fn mk_cycle<V: Debug>(c: &dyn Cache<V = V>) {
    // CHECK-LABEL: fn mk_cycle(
    // CHECK-NOT: inlined
    c.store_nocache()
}

// EMIT_MIR dyn_trait.try_execute_query.Inline.diff
#[inline(always)]
pub fn try_execute_query<C: Cache>(c: &C) {
    // CHECK-LABEL: fn try_execute_query(
    // CHECK: (inlined mk_cycle::<<C as Cache>::V>)
    mk_cycle(c)
}

// EMIT_MIR dyn_trait.get_query.Inline.diff
#[inline(always)]
pub fn get_query<Q: Query, T>(t: &T) {
    // CHECK-LABEL: fn get_query(
    // CHECK-NOT: inlined
    let c = Q::cache(t);
    // CHECK: (inlined try_execute_query::<<Q as Query>::C>)
    // CHECK: (inlined mk_cycle::<<Q as Query>::V>)
    try_execute_query(c)
}
