// Check that inline assembly diagnostics use the explicit register name from the source.
//
// On 32-bit x86, vector registers 8 and higher are unavailable. Registers 16 through 31
// share a canonical `zmm` representation, but diagnostics must still use the `xmm`, `ymm`,
// or `zmm` name that the user wrote.
//
// Regression test for <https://github.com/rust-lang/rust/issues/159409>.

//@ add-minicore
//@ revisions: x86 x86_64
//@[x86] compile-flags: --target i686-unknown-linux-gnu
//@[x86] needs-llvm-components: x86
//@[x86] check-fail
//@[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64] needs-llvm-components: x86
//@[x86_64] check-pass
//@ ignore-backends: gcc

#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

fn main() {
    unsafe {
        // Register 15 is the last register with separate canonical representations.
        asm!("", lateout("xmm15") _);
        //[x86]~^ ERROR cannot use register `xmm15`
        asm!("", lateout("ymm15") _);
        //[x86]~^ ERROR cannot use register `ymm15`
        asm!("", lateout("zmm15") _);
        //[x86]~^ ERROR cannot use register `zmm15`

        // Register 16 is the first register whose aliases share the canonical `zmm` representation.
        asm!("", lateout("xmm16") _);
        //[x86]~^ ERROR cannot use register `xmm16`
        asm!("", lateout("ymm16") _);
        //[x86]~^ ERROR cannot use register `ymm16`
        asm!("", lateout("zmm16") _);
        //[x86]~^ ERROR cannot use register `zmm16`

        // Register 31 is the upper boundary of the affected register range.
        asm!("", lateout("xmm31") _);
        //[x86]~^ ERROR cannot use register `xmm31`
        asm!("", lateout("ymm31") _);
        //[x86]~^ ERROR cannot use register `ymm31`
        asm!("", lateout("zmm31") _);
        //[x86]~^ ERROR cannot use register `zmm31`
    }
}
