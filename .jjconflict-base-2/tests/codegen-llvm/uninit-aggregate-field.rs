// Regression test for https://github.com/rust-lang/rust/issues/157743
//
// At opt-level >= 1, MIR GVN inlines MaybeUninit::uninit() and propagates the
// result as `const <uninit>` in aggregate constructions. Without the fix, this
// caused codegen to emit a memcpy from an `[N x i8] undef` global constant for
// the uninit field, which LLVM would materialize as zero-initialization.
//
// The fix skips emitting any IR for entirely-uninit constant aggregate fields.

//@ compile-flags: -C no-prepopulate-passes -C opt-level=2

#![crate_type = "lib"]

use std::mem::MaybeUninit;

pub struct Inner {
    cap: usize,
    data: MaybeUninit<[u64; 2]>,
}

// CHECK-LABEL: @make_inner
// The non-uninit field must be stored.
// CHECK: store i{{(32|64)}}
// The entirely-uninit `data` field must not cause a memcpy from an undef global.
// CHECK-NOT: call void @llvm.memcpy
// CHECK: ret void
#[no_mangle]
pub fn make_inner(cap: usize) -> Inner {
    Inner { cap, data: MaybeUninit::uninit() }
}
