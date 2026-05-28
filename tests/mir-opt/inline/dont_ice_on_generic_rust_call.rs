// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Zmir-enable-passes=+Inline --crate-type=lib

#![feature(fn_traits, tuple_trait, unboxed_closures)]

use std::marker::Tuple;

// EMIT_MIR dont_ice_on_generic_rust_call.call.Inline.diff
pub fn call<I: Tuple>(mut mock: Box<dyn FnMut<I, Output = ()>>, input: I) {
    // CHECK-LABEL: fn call(
    // CHECK: (inlined <Box<dyn FnMut<I, Output = ()>> as FnMut<I>>::call_mut)
    mock.call_mut(input)
}
