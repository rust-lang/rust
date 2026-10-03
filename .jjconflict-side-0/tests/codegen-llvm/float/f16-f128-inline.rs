//@ revisions: DEFAULT NOPT
//@[NOPT] compile-flags: -Copt-level=0 -Zcross-crate-inline-threshold=never -Zmir-opt-level=0 -Cno-prepopulate-passes

// Ensure that functions using `f16` and `f128` are always inlined when the backend does not
// support these types, to avoid crashes.

#![crate_type = "lib"]
#![feature(f128)]
#![feature(f16)]
#![feature(cfg_target_has_reliable_f16_f128)]

// This test does some tricky things. On `target_has_reliable_*` platforms:
//
// * `*_on_reliable` functions should always show up in codegen since they are not auto-inlined
//   (the default),
// * `*_on_not_reliable` functions aren't defined at all, so `CHECK-NOT` passes.
//
// On non-`target_has_reliable_*` platforms:
//
// * `*_on_reliable` functions are dummies so they always show up in codegen.
// * `*_on_not_reliable` functions should be auto-inlined and thus not show up in codegen.
//
// `*_on_reliable` is only checked with NOPT since otherwise they may hit auto-inlining thresholds
// unrelated to the type signature.

// NOPT: f16_arg_on_reliable
// NOPT: f16_ret_on_reliable
// CHECK-NOT: f16_arg_on_not_reliable
// CHECK-NOT: f16_ret_on_not_reliable
cfg_select! {
    target_has_reliable_f16 => {
        pub fn f16_arg_on_reliable(_a: f16) {
            todo!()
        }

        pub fn f16_ret_on_reliable() -> f16 {
            todo!()
        }
    }
    _ => {
        pub fn f16_arg_on_not_reliable(_a: f16) {
            todo!()
        }

        pub fn f16_ret_on_not_reliable() -> f16 {
            todo!()
        }

        #[unsafe(no_mangle)]
        pub fn f16_arg_on_reliable() {}
        #[unsafe(no_mangle)]
        pub fn f16_ret_on_reliable() {}
    }
}

// NOPT: f128_arg_on_reliable
// NOPT: f128_ret_on_reliable
// CHECK-NOT: f128_arg_on_not_reliable
// CHECK-NOT: f128_ret_on_not_reliable
cfg_select! {
    target_has_reliable_f128 => {
        pub fn f128_arg_on_reliable(_a: f128) {
            todo!()
        }

        pub fn f128_ret_on_reliable() -> f128 {
            todo!()
        }
    }
    _ => {
        pub fn f128_arg_on_not_reliable(_a: f128) {
            todo!()
        }

        pub fn f128_ret_on_not_reliable() -> f128 {
            todo!()
        }

        #[unsafe(no_mangle)]
        pub fn f128_arg_on_reliable() {}
        #[unsafe(no_mangle)]
        pub fn f128_ret_on_reliable() {}
    }
}
