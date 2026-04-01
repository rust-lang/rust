//@ add-minicore
//@ revisions: X86 AARCH64 UNSUPPORTED
//@ [X86] compile-flags: -C no-prepopulate-passes --target=x86_64-unknown-linux-gnu
//@ [X86] needs-llvm-components: x86
//@ [AARCH64] compile-flags: -C no-prepopulate-passes --target=aarch64-unknown-linux-gnu
//@ [AARCH64] needs-llvm-components: aarch64
//@ [UNSUPPORTED] compile-flags: -C no-prepopulate-passes --target=i686-unknown-linux-gnu
//@ [UNSUPPORTED] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(rust_preserve_none_cc)]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;

// X86: define{{( dso_local)?}} preserve_nonecc void @peach(i16
// AARCH64: define{{( dso_local)?}} preserve_nonecc void @peach(i16
// UNSUPPORTED: define{{( dso_local)?}} void @peach(i16
#[no_mangle]
#[inline(never)]
pub extern "rust-preserve-none" fn peach(x: u16) {
    loop {}
}

// X86: call preserve_nonecc void @peach(i16
// AARCH64: call preserve_nonecc void @peach(i16
// UNSUPPORTED: call void @peach(i16
pub fn quince(x: u16) {
    if let 12345u16 = x {
        peach(54321);
    }
}
