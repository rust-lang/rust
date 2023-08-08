// compile-flags: --crate-type=lib
// unit-test: InstSimplify

#![feature(core_intrinsics)]

// Want to make sure this assertion isn't compiled away in generic code.

// EMIT_MIR dont_yeet_assert.generic.InstSimplify.diff
pub fn generic<T>() {
    core::intrinsics::assert_mem_uninitialized_valid::<&T>();
}
