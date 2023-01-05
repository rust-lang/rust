// revisions: ropi rwpi

// [ropi] compile-flags: --target armv7-unknown-linux-gnueabihf -C relocation-model=ropi
// [rwpi] compile-flags: --target armv7-unknown-linux-gnueabihf -C relocation-model=rwpi
// [ropi] needs-llvm-components: arm
// [rwpi] needs-llvm-components: arm
// [ropi] build-pass

#![feature(no_core, lang_items, rustc_attrs)]
#![no_core]
#![crate_type = "rlib"]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}
#[lang = "sized"]
trait Sized {}

// R9 is reserved as the RWPI base register
fn main() {
    unsafe {
        asm!("", out("r9") _);
        //[rwpi]~^ cannot use register `r9`
    }
}
