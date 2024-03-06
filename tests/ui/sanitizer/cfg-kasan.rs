// Verifies that when compiling with -Zsanitizer=kernel-address,
// the `#[cfg(sanitize = "address")]` attribute is configured.

//@ check-pass
//@ compile-flags: -Zsanitizer=kernel-address --cfg kernel_address
//@ revisions: aarch64 riscv64imac riscv64gc x86_64
//@[aarch64] compile-flags: --target aarch64-unknown-none
//@[aarch64] needs-llvm-components: aarch64
//@[riscv64imac] compile-flags: --target riscv64imac-unknown-none-elf
//@[riscv64imac] needs-llvm-components: riscv
//@[riscv64gc] compile-flags: --target riscv64gc-unknown-none-elf
//@[riscv64gc] needs-llvm-components: riscv
//@[x86_64] compile-flags: --target x86_64-unknown-none
//@[x86_64] needs-llvm-components: x86

#![crate_type = "rlib"]
#![feature(cfg_sanitize, no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

const _: fn() -> () = main;

#[cfg(all(sanitize = "address", kernel_address))]
fn main() {}
