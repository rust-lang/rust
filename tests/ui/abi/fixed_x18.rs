// This tests that -Zfixed-x18 causes a compilation failure on targets other than aarch64.
// Behavior on aarch64 is tested by tests/codegen-llvm/fixed-x18.rs.
//
//@ revisions: x64 i686 arm riscv32 riscv64
//@ dont-check-compiler-stderr
//
//@ compile-flags: -Zfixed-x18
//@ [x64] needs-llvm-components: x86
//@ [x64] compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib
//@ [i686] needs-llvm-components: x86
//@ [i686] compile-flags: --target=i686-unknown-linux-gnu --crate-type=rlib
//@ [arm] needs-llvm-components: arm
//@ [arm] compile-flags: --target=armv7-unknown-linux-gnueabihf --crate-type=rlib
//@ [riscv32] needs-llvm-components: riscv
//@ [riscv32] compile-flags: --target=riscv32i-unknown-none-elf --crate-type=rlib
//@ [riscv64] needs-llvm-components: riscv
//@ [riscv64] compile-flags: --target=riscv64gc-unknown-none-elf --crate-type=rlib

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointee_sized"]
trait PointeeSized {}
#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}
#[lang = "sized"]
trait Sized: MetaSized {}

//~? ERROR the `-Zfixed-x18` flag is not supported on the `
