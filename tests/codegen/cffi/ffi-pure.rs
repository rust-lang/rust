//@ compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(ffi_pure)]

pub fn bar() { unsafe { foo() } }

extern "C" {
    // CHECK-LABEL: declare{{.*}}void @foo()
    // CHECK-SAME: [[ATTRS:#[0-9]+]]
    // The attribute changed from `readonly` to `memory(read)` with LLVM 16.0.
    // CHECK-DAG: attributes [[ATTRS]] = { {{.*}}{{readonly|memory\(read\)}}{{.*}} }
    #[ffi_pure] pub fn foo();
}
