//@ build-pass
//@ ignore-backends: gcc
//@ add-minicore
//@ min-llvm-version: 22
//
//@ revisions: i686
//@[i686] compile-flags: --target i686-unknown-linux-gnu
//@[i686] needs-llvm-components: x86
//@ revisions: x86-64
//@[x86-64] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86-64] needs-llvm-components: x86
//@ revisions: x86-64-win
//@[x86-64-win] compile-flags: --target x86_64-pc-windows-msvc
//@[x86-64-win] needs-llvm-components: x86
//@ revisions: arm
//@[arm] compile-flags: --target arm-unknown-linux-gnueabi
//@[arm] needs-llvm-components: arm
//@ revisions: thumb
//@[thumb] compile-flags: --target thumbv8m.main-none-eabi
//@[thumb] needs-llvm-components: arm
//@ revisions: aarch64
//@[aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64] needs-llvm-components: aarch64
//@ revisions: s390x
//@[s390x] compile-flags: --target s390x-unknown-linux-gnu
//@[s390x] needs-llvm-components: systemz
//@ revisions: sparc
//@[sparc] compile-flags: --target sparc-unknown-linux-gnu
//@[sparc] needs-llvm-components: sparc
//@ revisions: sparc64
//@[sparc64] compile-flags: --target sparc64-unknown-linux-gnu
//@[sparc64] needs-llvm-components: sparc
//@ revisions: powerpc64
//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//@[powerpc64] needs-llvm-components: powerpc
//@ revisions: loongarch32
//@[loongarch32] compile-flags: --target loongarch32-unknown-none
//@[loongarch32] needs-llvm-components: loongarch
//@[loongarch32] ignore-llvm-version: 22 - 23
//@ revisions: loongarch64
//@[loongarch64] compile-flags: --target loongarch64-unknown-linux-gnu
//@[loongarch64] needs-llvm-components: loongarch
//@[loongarch64] ignore-llvm-version: 22 - 23
//@ revisions: bpf
//@[bpf] compile-flags: --target bpfeb-unknown-none
//@[bpf] needs-llvm-components: bpf
//@ revisions: m68k
//@[m68k] compile-flags: --target m68k-unknown-linux-gnu
//@[m68k] needs-llvm-components: m68k
//@ revisions: nvptx64
//@[nvptx64] compile-flags: --target nvptx64-nvidia-cuda
//@[nvptx64] needs-llvm-components: nvptx
//
// Riscv does not support byval in LLVM 22 (but wil in LLVM 23)
//
// //@ revisions: riscv
// //@[riscv] compile-flags: --target riscv64gc-unknown-linux-gnu
// //@[riscv] needs-llvm-components: riscv
//
// Wasm needs a special target feature.
//
//@ revisions: wasm
//@[wasm] compile-flags: --target wasm32-unknown-unknown -Ctarget-feature=+tail-call
//@[wasm] needs-llvm-components: webassembly
//@ revisions: wasip1
//@[wasip1] compile-flags: --target wasm32-wasip1 -Ctarget-feature=+tail-call
//@[wasip1] needs-llvm-components: webassembly
//
// Failing cases (just zero support)
//
// //@ revisions: powerpc
// //@[powerpc] compile-flags: --target powerpc-unknown-linux-gnu
// //@[powerpc] needs-llvm-components: powerpc
// //@ revisions: aix
// //@[aix] compile-flags: --target powerpc64-ibm-aix
// //@[aix] needs-llvm-components: powerpc
// //@ revisions: csky
// //@[csky] compile-flags: --target csky-unknown-linux-gnuabiv2
// //@[csky] needs-llvm-components: csky
// //@ revisions: mips
// //@[mips] compile-flags: --target mips-unknown-linux-gnu
// //@[mips] needs-llvm-components: mips
// //@ revisions: mips64
// //@[mips64] compile-flags: --target mips64-unknown-linux-gnuabi64
// //@[mips64] needs-llvm-components: mips
#![feature(no_core, explicit_tail_calls)]
#![expect(incomplete_features)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[repr(C)]
struct PassedByVal {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
}

#[inline(never)]
extern "C" fn callee(x: PassedByVal) -> PassedByVal {
    x
}

#[unsafe(no_mangle)]
extern "C" fn byval(x: PassedByVal) -> PassedByVal {
    become callee(x);
}
