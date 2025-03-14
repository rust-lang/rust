//@ add-core-stubs
//@ needs-llvm-components: riscv
//@ compile-flags: --target=riscv32imc-unknown-none-elf --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;
use minicore::*;

// Test that the riscv interrupt ABIs cannot be used when riscv_interrupt
// feature gate is not used.

extern "riscv-interrupt-m" fn f() {}
//~^ ERROR "riscv-interrupt-m" ABI is experimental
extern "riscv-interrupt-s" fn f_s() {}
//~^ ERROR "riscv-interrupt-s" ABI is experimental

trait T {
    extern "riscv-interrupt-m" fn m();
    //~^ ERROR "riscv-interrupt-m" ABI is experimental
}

struct S;
impl T for S {
    extern "riscv-interrupt-m" fn m() {}
    //~^ ERROR "riscv-interrupt-m" ABI is experimental
}

impl S {
    extern "riscv-interrupt-m" fn im() {}
    //~^ ERROR "riscv-interrupt-m" ABI is experimental
}

type TA = extern "riscv-interrupt-m" fn();
//~^ ERROR "riscv-interrupt-m" ABI is experimental
