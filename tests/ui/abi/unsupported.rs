// revisions: x64 i686 aarch64 arm
//
// [x64] needs-llvm-components: x86
// [x64] compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib
// [i686] needs-llvm-components: x86
// [i686] compile-flags: --target=i686-unknown-linux-gnu --crate-type=rlib
// [aarch64] needs-llvm-components: aarch64
// [aarch64] compile-flags: --target=aarch64-unknown-linux-gnu --crate-type=rlib
// [arm] needs-llvm-components: arm
// [arm] compile-flags: --target=armv7-unknown-linux-gnueabihf --crate-type=rlib
#![no_core]
#![feature(
    no_core,
    lang_items,
    abi_ptx,
    abi_msp430_interrupt,
    abi_avr_interrupt,
    abi_thiscall,
    abi_amdgpu_kernel,
    wasm_abi,
    abi_x86_interrupt
)]
#[lang="sized"]
trait Sized { }

extern "ptx-kernel" fn ptx() {}
//~^ ERROR is not a supported ABI
extern "amdgpu-kernel" fn amdgpu() {}
//~^ ERROR is not a supported ABI
extern "wasm" fn wasm() {}
//~^ ERROR is not a supported ABI
extern "aapcs" fn aapcs() {}
//[x64]~^ ERROR is not a supported ABI
//[i686]~^^ ERROR is not a supported ABI
//[aarch64]~^^^ ERROR is not a supported ABI
extern "msp430-interrupt" fn msp430() {}
//~^ ERROR is not a supported ABI
extern "avr-interrupt" fn avr() {}
//~^ ERROR is not a supported ABI
extern "x86-interrupt" fn x86() {}
//[aarch64]~^ ERROR is not a supported ABI
//[arm]~^^ ERROR is not a supported ABI
extern "thiscall" fn thiscall() {}
//[x64]~^ ERROR is not a supported ABI
//[aarch64]~^^ ERROR is not a supported ABI
//[arm]~^^^ ERROR is not a supported ABI
extern "stdcall" fn stdcall() {}
//[x64]~^ WARN use of calling convention not supported
//[x64]~^^ WARN this was previously accepted
//[aarch64]~^^^ WARN use of calling convention not supported
//[aarch64]~^^^^ WARN this was previously accepted
//[arm]~^^^^^ WARN use of calling convention not supported
//[arm]~^^^^^^ WARN this was previously accepted
