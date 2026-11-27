// Check that DIFlagAllCallsDescribed is set on subprogram definitions
// in optimized builds, so LLVM emits DW_TAG_call_site entries.

//@ ignore-msvc (CodeView does not use DIFlagAllCallsDescribed)
//@ compile-flags: -C debuginfo=2 -C opt-level=1 -C no-prepopulate-passes

#![crate_type = "lib"]

// CHECK: {{.*}}DISubprogram{{.*}}name: "foo"{{.*}}DIFlagAllCallsDescribed{{.*}}

#[no_mangle]
#[inline(never)]
pub fn foo(x: i32) -> i32 {
    bar(x + 1)
}

#[no_mangle]
#[inline(never)]
pub fn bar(x: i32) -> i32 {
    x * 2
}
