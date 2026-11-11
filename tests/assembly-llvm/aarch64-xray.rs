//@ assembly-output: emit-asm
//@ compile-flags: -Zinstrument-xray=always

//@ revisions: aarch64-linux
//@[aarch64-linux] compile-flags: --target=aarch64-unknown-linux-gnu
//@[aarch64-linux] needs-llvm-components: aarch64
//@[aarch64-linux] only-aarch64-unknown-linux-gnu

//@ revisions: aarch64-darwin
//@[aarch64-darwin] compile-flags: --target=aarch64-apple-darwin
//@[aarch64-darwin] needs-llvm-components: aarch64
//@[aarch64-darwin] only-aarch64-apple-darwin

#![crate_type = "lib"]

// CHECK-LABEL: xray_func:
#[no_mangle]
pub fn xray_func() {
    // CHECK: nop

    std::hint::black_box(());

    // CHECK: b #32
    // CHECK-NEXT: nop
}
