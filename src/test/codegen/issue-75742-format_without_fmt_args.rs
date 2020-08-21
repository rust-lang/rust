// min-llvm-version: 10.0.0
// compile-flags: -C opt-level=3
#![crate_type = "rlib"]
#![feature(format_args_capture)]

// Make sure allocation not happen when result of `format!` is unused and
// there are no formatting arguments.

// CHECK-LABEL: @format_wo_fmt_args
// CHECK-NEXT: {{"_ZN[^:]+"}}:
// CHECK-NEXT: ret
#[no_mangle]
pub fn format_wo_fmt_args() {
    format!("");
    format!("a long story");
    format!("a long story {{");
}

// CHECK-LABEL: @format_wo_fmt_args_ret
// CHECK-NOT: Arguments
#[no_mangle]
pub fn format_wo_fmt_args_ret() -> String {
    format!("a long story")
}

// CHECK-LABEL: @format_w_fmt_args_ret_1
// CHECK: alloc::fmt::format
#[no_mangle]
pub fn format_w_fmt_args_ret_1(n: usize) -> String {
    format!("a long story: {}", n)
}

// CHECK-LABEL: @format_w_fmt_args_ret_2
// CHECK: core::fmt::ArgumentV1::from_usize
// CHECK: alloc::fmt::format
#[no_mangle]
pub fn format_w_fmt_args_ret_2(n: usize, width: usize) -> String {
    format!("a long story {n:width$}")
}
