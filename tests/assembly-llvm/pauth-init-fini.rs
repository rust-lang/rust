// ignore-tidy-file-linelength
//@ add-minicore
//@ assembly-output: emit-asm
//@ revisions: ADDR_DISC NO_ADDR_DISC
//@ [ADDR_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest
//@ [ADDR_DISC] needs-llvm-components: aarch64
//@ [NO_ADDR_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest -Zpointer-authentication=-init-fini-address-discrimination
//@ [NO_ADDR_DISC] needs-llvm-components: aarch64

// Init/fini entries are signed with the init/fini discriminator (55764). By default they are also
// address-discriminated, which LLVM emits as `addr` in the AUTH relocation: the loader blends in
// the address of the entry.
// `-init-fini-address-discrimination` turns that off.

#![feature(no_core)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

extern "C" fn init_fn() {}

// CHECK: .section .init_array.90
// ADDR_DISC: .xword {{.*}}init_fn@AUTH(ia,55764,addr)
// NO_ADDR_DISC: .xword {{.*}}init_fn@AUTH(ia,55764){{$}}
#[used]
#[link_section = ".init_array.90"]
static GLOBAL_INIT: extern "C" fn() = init_fn;
