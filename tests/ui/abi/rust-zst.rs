//@ add-minicore
//@ normalize-stderr: "(align|unadjusted_abi_align): Align\([1-8] bytes\)" -> "$1: $$SOME_ALIGN"
/*!
Rust ZSTs are always ignored in the calling convention even when the C ABI
passes zero-sized structs by-ref.

That means these two function signatures are always compatible:

```
extern "C" fn((), i32, i32);
extern "C" fn(i32, (), i32);
```
*/

//
// C ZST IS ZERO-SIZED
//

//@ revisions: aarch64-darwin
//@[aarch64-darwin] compile-flags: --target aarch64-apple-darwin
//@[aarch64-darwin] needs-llvm-components: aarch64

//@ revisions: x86_64-linux
//@[x86_64-linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64-linux] needs-llvm-components: x86

//
// C ZST IS PASS-BY-POINTER
//

// according to the SRV4 ABI, an aggregate is always passed in registers,
// and it so happens the GCC extension for ZSTs considers them as structs.
//@ revisions: powerpc-linux
//@[powerpc-linux] compile-flags: --target powerpc-unknown-linux-gnu
//@[powerpc-linux] needs-llvm-components: powerpc

//@ revisions: s390x-linux
//@[s390x-linux] compile-flags: --target s390x-unknown-linux-gnu
//@[s390x-linux] needs-llvm-components: systemz

//@ revisions: sparc-none sparc-linux sparc64-linux
//@[sparc-none] compile-flags: --target sparc-unknown-none-elf
//@[sparc-none] needs-llvm-components: sparc
//@[sparc-linux] compile-flags: --target sparc-unknown-linux-gnu
//@[sparc-linux] needs-llvm-components: sparc
//@[sparc64-linux] compile-flags: --target sparc64-unknown-linux-gnu
//@[sparc64-linux] needs-llvm-components: sparc

// The Win64 ABI uses slightly different handling for power-of-2 sizes in the ABI,
// so GCC decided that ZSTs are pass-by-pointer, as `0.is_power_of_two() == false`
//@ revisions: x86_64-pc-windows-gnu
//@[x86_64-pc-windows-gnu] compile-flags: --target x86_64-pc-windows-gnu
//@[x86_64-pc-windows-gnu] needs-llvm-components: x86
//@ ignore-backends: gcc

#![feature(no_core, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[rustc_abi(debug)]
extern "C" fn pass_rust_zst(_: ()) {} //~ ERROR: fn_abi
