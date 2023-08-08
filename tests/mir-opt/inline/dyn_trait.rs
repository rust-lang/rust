// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
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
    c.store_nocache()
}

// EMIT_MIR dyn_trait.try_execute_query.Inline.diff
#[inline(always)]
pub fn try_execute_query<C: Cache>(c: &C) {
    mk_cycle(c)
}

// EMIT_MIR dyn_trait.get_query.Inline.diff
#[inline(always)]
pub fn get_query<Q: Query, T>(t: &T) {
    let c = Q::cache(t);
    try_execute_query(c)
}
