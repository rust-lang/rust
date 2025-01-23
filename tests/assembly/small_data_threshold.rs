// Test for -Z small_data_threshold=...
//@ revisions: RISCV MIPS HEXAGON M68K
//@ assembly-output: emit-asm
//@ compile-flags: -Z small_data_threshold=4
//@ [RISCV] compile-flags: --target=riscv32im-unknown-none-elf
//@ [RISCV] needs-llvm-components: riscv
//@ [MIPS] compile-flags: --target=mips-unknown-linux-uclibc -C relocation-model=static
//@ [MIPS] compile-flags: -C llvm-args=-mgpopt -C llvm-args=-mlocal-sdata
//@ [MIPS] compile-flags: -C target-feature=+noabicalls
//@ [MIPS] needs-llvm-components: mips
//@ [HEXAGON] compile-flags: --target=hexagon-unknown-linux-musl -C target-feature=+small-data
//@ [HEXAGON] compile-flags: -C llvm-args=--hexagon-statics-in-small-data
//@ [HEXAGON] needs-llvm-components: hexagon
//@ [M68K] compile-flags: --target=m68k-unknown-linux-gnu
//@ [M68K] needs-llvm-components: m68k

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[lang = "drop_in_place"]
fn drop_in_place<T>(_: *mut T) {}

#[used]
#[no_mangle]
// U is below the threshold, should be in sdata
static mut U: u16 = 123;

#[used]
#[no_mangle]
// V is below the threshold, should be in sbss
static mut V: u16 = 0;

#[used]
#[no_mangle]
// W is at the threshold, should be in sdata
static mut W: u32 = 123;

#[used]
#[no_mangle]
// X is at the threshold, should be in sbss
static mut X: u32 = 0;

#[used]
#[no_mangle]
// Y is over the threshold, should be in its own .data section
static mut Y: u64 = 123;

#[used]
#[no_mangle]
// Z is over the threshold, should be in its own .bss section
static mut Z: u64 = 0;

// Currently, only MIPS and RISCV successfully put any objects in the small data
// sections so the U/V/W/X tests are skipped on Hexagon and M68K

// RISCV: .section .sdata
// RISCV-NOT: .section
// RISCV: U:
// RISCV: .section .sbss
// RISCV-NOT: .section
// RISCV: V:
// RISCV: .section .sdata
// RISCV-NOT: .section
// RISCV: W:
// RISCV: .section .sbss
// RISCV-NOT: .section
// RISCV: X:

// MIPS: .section .sdata
// MIPS-NOT: .section
// MIPS: U:
// MIPS: .section .sbss
// MIPS-NOT: .section
// MIPS: V:
// MIPS: .section .sdata
// MIPS-NOT: .section
// MIPS: W:
// MIPS: .section .sbss
// MIPS-NOT: .section
// MIPS: X:

// CHECK: .section .data.Y,
// CHECK-NOT: .section
// CHECK: Y:
// CHECK: .section .bss.Z,
// CHECK-NOT: .section
// CHECK: Z:
