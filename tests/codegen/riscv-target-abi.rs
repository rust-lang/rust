//@ add-core-stubs
//@ revisions:riscv64gc riscv32gc riscv32imac

//@[riscv64gc] compile-flags: --target=riscv64gc-unknown-linux-gnu
//@[riscv64gc] needs-llvm-components: riscv
// riscv64gc: !{i32 1, !"target-abi", !"lp64d"}

//@[riscv32gc] compile-flags: --target=riscv32gc-unknown-linux-musl
//@[riscv32gc] needs-llvm-components: riscv
// riscv32gc: !{i32 1, !"target-abi", !"ilp32d"}

//@[riscv32imac] compile-flags: --target=riscv32imac-unknown-none-elf
//@[riscv32imac] needs-llvm-components: riscv
// riscv32imac: !{i32 1, !"target-abi", !"ilp32"}

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;
