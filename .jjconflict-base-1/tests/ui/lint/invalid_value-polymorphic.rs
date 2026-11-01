//@ compile-flags: --crate-type=lib -Zmir-enable-passes=+InstSimplify-before-inline
//@ build-pass

#![feature(core_intrinsics)]

pub fn generic<T>() {
    core::intrinsics::assert_mem_uninitialized_valid::<&T>();
}
