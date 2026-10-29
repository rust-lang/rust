// Check that DIFlagAllCallsDescribed is NOT set in unoptimized builds.
// At -O0 no tail-call optimization occurs, so the debugger can
// reconstruct the call stack without DW_TAG_call_site entries.

//@ ignore-msvc (CodeView does not use DIFlagAllCallsDescribed)
//@ compile-flags: -C debuginfo=2 -C opt-level=0 -C no-prepopulate-passes

#![crate_type = "lib"]

// CHECK: {{.*}}DISubprogram{{.*}}name: "foo"
// CHECK-NOT: DIFlagAllCallsDescribed

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
