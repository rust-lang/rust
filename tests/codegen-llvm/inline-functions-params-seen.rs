// Checks that the LLVM backend's `CodegenUnitDebugContext::params_seen` hashmap is working.

// Note: -Copt-level=1 ensures the required `ReferencePropagation` pass runs.
//@ compile-flags: -Zinline-mir=yes -Copt-level=1 -Cdebuginfo=2 -Ccodegen-units=1 -Zverify-llvm-ir
//@ edition: 2021

#![crate_type = "lib"]

extern "Rust" {
    fn sink(x: &u8);
}

#[inline]
fn callee(x: &u8) {
    // Check the following:
    // - No `define` for `callee` (because it's inlined).
    // - Exactly one `x` variable entry.
    //
    // CHECK-NOT: define{{.*}}callee
    // CHECK: !DILocalVariable(name: "x", arg: 1
    // CHECK-NOT: !DILocalVariable(name: "x", arg: 1
    unsafe { sink(x) }
}

pub fn f1(s1: &u8, s2: &mut u8, s3: *const u8) {
    // Three different types are needed here so that LLVM doesn't merge the debuginfo variable
    // entries itself.
    callee(s1);
    callee(&*s2);
    callee(unsafe { &*s3 });
}
