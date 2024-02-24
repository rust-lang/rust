//@ assembly-output: emit-asm
//@ only-x86_64-unknown-linux-gnu
//@ compile-flags: -C panic=unwind -C force-unwind-tables=n

#![crate_type = "lib"]

// CHECK-LABEL: foo:
// CHECK: .cfi_startproc
#[no_mangle]
fn foo() {
    panic!();
}
