//@ revisions: default nopt
//@[nopt] compile-flags: -Copt-level=0 -Zcross-crate-inline-threshold=never -Zmir-opt-level=0 -Cno-prepopulate-passes

// Ensure that functions using `f16` and `f128` are always inlined to avoid crashes
// when the backend does not support these types.

#![crate_type = "lib"]
#![feature(f128)]
#![feature(f16)]

pub fn f16_arg(_a: f16) {
    // CHECK-NOT: f16_arg
    todo!()
}

pub fn f16_ret() -> f16 {
    // CHECK-NOT: f16_ret
    todo!()
}

pub fn f128_arg(_a: f128) {
    // CHECK-NOT: f128_arg
    todo!()
}

pub fn f128_ret() -> f128 {
    // CHECK-NOT: f128_ret
    todo!()
}
