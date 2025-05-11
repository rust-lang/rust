//@ add-core-stubs
//@ revisions: msp430 aarch32
//@[msp430] needs-llvm-components: msp430
//@[msp430] compile-flags: --target=msp430-none-elf
//@[aarch32] build-pass
//@[aarch32] needs-llvm-components: arm
//@[aarch32] compile-flags: --target=thumbv7m-none-eabi

// We should fail to compute alignment for types aligned higher than usize::MAX.
// We can't handle alignments that require all 32 bits, so this only affects 16-bit.

#![feature(lang_items, no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[repr(align(16384))]
struct Kitten;

#[repr(align(32768))] //[msp430]~ ERROR alignment must not be greater than `isize::MAX`
struct Cat;

#[repr(align(65536))] //[msp430]~ ERROR alignment must not be greater than `isize::MAX`
struct BigCat;
