//@ add-minicore
//@ revisions: I586 X86_64 AARCH64
//@ [I586] compile-flags: -C no-prepopulate-passes --target=i586-unknown-linux-gnu
//@ [I586] needs-llvm-components: x86
//@ [X86_64] compile-flags: -C no-prepopulate-passes --target=x86_64-unknown-linux-gnu
//@ [X86_64] needs-llvm-components: x86
//@ [AARCH64] compile-flags: -C no-prepopulate-passes --target=aarch64-unknown-linux-gnu
//@ [AARCH64] needs-llvm-components: aarch64

#![crate_type = "lib"]
#![feature(no_core, rust_tail_cc, explicit_tail_calls)]
#![no_core]

extern crate minicore;

// CHECK: define{{( dso_local)?}} tailcc void @peach(i16
#[no_mangle]
#[inline(never)]
pub extern "tail" fn peach(_: u16) {
    loop {}
}

// CHECK: call tailcc void @peach(i16
pub fn quince(x: u16) {
    if let 12345u16 = x {
        peach(54321);
    }
}
