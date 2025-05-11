//@ add-core-stubs
//@ revisions: ropi rwpi

//@ [ropi] compile-flags: --target armv7-unknown-linux-gnueabihf -C relocation-model=ropi
//@ [rwpi] compile-flags: --target armv7-unknown-linux-gnueabihf -C relocation-model=rwpi
//@ [ropi] needs-llvm-components: arm
//@ [rwpi] needs-llvm-components: arm
//@ [ropi] build-pass

#![feature(no_core)]
#![no_core]
#![crate_type = "rlib"]

extern crate minicore;
use minicore::*;

// R9 is reserved as the RWPI base register
fn main() {
    unsafe {
        asm!("", out("r9") _);
        //[rwpi]~^ ERROR cannot use register `r9`
    }
}
