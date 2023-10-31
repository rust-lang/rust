// compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(ffi_const)]

pub fn bar() { unsafe { foo() } }

extern "C" {
    // CHECK-LABEL: declare{{.*}}void @foo()
    // CHECK-SAME: [[ATTRS:#[0-9]+]]
    // The attribute changed from `readnone` to `memory(none)` with LLVM 16.0.
    // CHECK-DAG: attributes [[ATTRS]] = { {{.*}}{{readnone|memory\(none\)}}{{.*}} }
    #[ffi_const] pub fn foo();
}
