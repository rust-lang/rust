//@ add-minicore
//@ ignore-backends: gcc
//
//@ revisions: riscv32e sparc avr m68k msp430
//
//@[riscv32e] compile-flags: --target riscv32e-unknown-none-elf
//@[riscv32e] needs-llvm-components: riscv
//
//@[sparc] compile-flags: --target sparc-unknown-none-elf
//@[sparc] needs-llvm-components: sparc
//
//@[avr] compile-flags: --target avr-none -Ctarget-cpu=atmega328p
//@[avr] needs-llvm-components: avr
//
//@[m68k] compile-flags: --target m68k-unknown-none-elf -Ctarget-cpu=M68020
//@[m68k] needs-llvm-components: m68k
//
//@[msp430] compile-flags: --target msp430-none-elf -Ctarget-cpu=msp430
//@[msp430] needs-llvm-components: msp430
#![feature(no_core, lang_items, rustc_attrs, c_variadic)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(transparent)]
struct VaListInner {
    ptr: *const c_void,
}

#[repr(transparent)]
#[lang = "va_list"]
pub struct VaList<'a> {
    inner: VaListInner,
    _marker: PhantomData<&'a mut ()>,
}

pub unsafe extern "C" fn test(_: i32, ap: ...) {}
//~^ ERROR C-variadic function definitions on this target are unstable

trait Trait {
    unsafe extern "C" fn trait_test(_: i32, ap: ...) {}
    //~^ ERROR C-variadic function definitions on this target are unstable
}
