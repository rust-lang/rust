//! Verify that simple intrinsics emit the instructions we expect.

//@ add-minicore
//@ assembly-output: emit-asm
//@ revisions: AARCH64 ARM64EC ARM32 AVR BPF LOONGARCH32 LOONGARCH64 I686 M68K MIPS32 MIPS64 MSP430 PPC32 PPC64 RISCV32 RISCV64 S390X SPARC32 SPARC64 WASM32 WASI32 X86_64 XTENSA
//
//@ [AARCH64] compile-flags: --target=aarch64-unknown-linux-gnu
//@ [AARCH64] needs-llvm-components: aarch64
//@ [ARM64EC] compile-flags: --target=arm64ec-pc-windows-msvc
//@ [ARM64EC] filecheck-flags: --check-prefixes AARCH64
//@ [ARM64EC] needs-llvm-components: aarch64
//
//@ [BPF] compile-flags: --target=bpfel-unknown-none
//@ [BPF] needs-llvm-components: bpf
//
//@ [AVR] compile-flags: --target=avr-none -Ctarget-cpu=atmega328p
//@ [AVR] needs-llvm-components: avr
//
//@ [ARM32] compile-flags: --target=armv7-unknown-linux-gnueabi
//@ [ARM32] needs-llvm-components: arm
//
//@ [LOONGARCH32] compile-flags: --target=loongarch32-unknown-none
//@ [LOONGARCH32] filecheck-flags: --check-prefixes LOONGARCH
//@ [LOONGARCH32] needs-llvm-components: loongarch
//@ [LOONGARCH64] compile-flags: --target=loongarch64-unknown-linux-gnu
//@ [LOONGARCH64] filecheck-flags: --check-prefixes LOONGARCH
//@ [LOONGARCH64] needs-llvm-components: loongarch
//@ [LOONGARCH32] min-llvm-version: 22
//@ [LOONGARCH64] min-llvm-version: 22
//
//@ [M68K] compile-flags: --target=m68k-unknown-linux-gnu
//@ [M68K] needs-llvm-components: m68k
//
//@ [MIPS32] compile-flags: --target=mips-unknown-linux-gnu
//@ [MIPS32] filecheck-flags: --check-prefixes MIPS
//@ [MIPS32] needs-llvm-components: mips
//@ [MIPS64] compile-flags: --target=mips64-unknown-linux-gnuabi64
//@ [MIPS64] filecheck-flags: --check-prefixes MIPS
//@ [MIPS64] needs-llvm-components: mips
//
//@ [MSP430] compile-flags: --target=msp430-none-elf
//@ [MSP430] needs-llvm-components: msp430
//
//@ [PPC32] compile-flags: --target=powerpc-unknown-linux-gnu
//@ [PPC32] filecheck-flags: --check-prefixes PPC
//@ [PPC32] needs-llvm-components: powerpc
//@ [PPC64] compile-flags: --target=powerpc64-unknown-linux-gnu
//@ [PPC64] filecheck-flags: --check-prefixes PPC
//@ [PPC64] needs-llvm-components: powerpc
//
//@ [RISCV32] compile-flags: --target=riscv32im-unknown-none-elf
//@ [RISCV32] filecheck-flags: --check-prefixes RISCV
//@ [RISCV32] needs-llvm-components: riscv
//@ [RISCV64] compile-flags: --target=riscv64gc-unknown-linux-gnu
//@ [RISCV64] filecheck-flags: --check-prefixes RISCV
//@ [RISCV64] needs-llvm-components: riscv
//
//@ [S390X] compile-flags: --target=s390x-unknown-linux-gnu
//@ [S390X] needs-llvm-components: systemz
//
//@ [SPARC32] compile-flags: --target=sparc-unknown-linux-gnu
//@ [SPARC32] filecheck-flags: --check-prefixes SPARC
//@ [SPARC32] needs-llvm-components: sparc
//@ [SPARC64] compile-flags: --target=sparc64-unknown-linux-gnu
//@ [SPARC64] filecheck-flags: --check-prefixes SPARC
//@ [SPARC64] needs-llvm-components: sparc
//
//@ [WASM32] compile-flags: --target=wasm32-unknown-unknown
//@ [WASM32] filecheck-flags: --check-prefixes WASM
//@ [WASM32] needs-llvm-components: webassembly
//@ [WASI32] compile-flags: --target=wasm32-wasip3
//@ [WASI32] filecheck-flags: --check-prefixes WASM
//@ [WASI32] needs-llvm-components: webassembly
//
//@ [I686] compile-flags: --target=i686-unknown-linux-gnu
//@ [I686] filecheck-flags: --check-prefixes X86
//@ [I686] needs-llvm-components: x86
//@ [X86_64] compile-flags: --target=x86_64-unknown-linux-gnu
//@ [X86_64] filecheck-flags: --check-prefixes X86
//@ [X86_64] needs-llvm-components: x86
//
//@ [XTENSA] compile-flags: --target=xtensa-esp32-espidf
//@ [XTENSA] needs-llvm-components: xtensa
//@ [XTENSA] min-llvm-version: 23

#![crate_type = "lib"]
#![feature(no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]

extern crate minicore;

mod intrinsics {
    #[rustc_intrinsic]
    #[rustc_nounwind]
    pub fn abort() -> !;

    #[rustc_intrinsic]
    #[rustc_nounwind]
    pub fn breakpoint();
}

// Optional `"` since labels are quoted on arm64ec. Forbid calls by default since most platforms
// shouldn't be calling out to libc for this intrinsic.
//
// Note that some platforms (MSP430, AVR) do require an abort builtin, but this is typically a
// loop or hardware reset.
//
// We don't make a user-facing guarantee about what exactly `abort` will do but it also shouldn't
// change.
//
// CHECK-LABEL: do_abort{{"?}}:
// CHECK-NOT: {{ (call|b|bl) }}
//
// AARCH64: brk #0x1
// AVR: call abort
// BPF: call __bpf_trap
// ARM32: .inst   0xe7ffdefe
// LOONGARCH: ud 0
// M68K: jsr (abort@PLT,%pc)
// MIPS: break
// MSP430: call #abort
// PPC: trap
// RISCV: unimp
// // S390x lowers to a loop
// S390X: .Ltmp{{.*}}:
// S390X: j .Ltmp{{.*}}
// SPARC: ta 5
// WASM: unreachable
// X86: ud2
// XTENSA: ill.n
#[unsafe(no_mangle)]
pub fn do_abort() {
    intrinsics::abort();
}

// Optional `"` since labels are quoted on arm64ec. Similar to `abort`, there typically shouldn't
// be any libcalls here.
//
// CHECK-LABEL: do_breakpoint{{"?}}:
// CHECK-NOT: {{ (call|b|bl|j) }}
//
// Documentation guarantees these instructions:
//
// AARCH64: brk #0xf000
// X86: int3
//
// We do not guarantee exact output on the following platforms but still check them:
//
// ARM32: bkpt #0
// AVR: call abort
// BPF: call __bpf_trap
// LOONGARCH: break 0
// M68K: jsr (abort@PLT,%pc)
// MIPS: break
// MSP430: call #abort
// PPC: trap
// RISCV: ebreak
// // S390x lowers to a loop
// S390X: .Ltmp{{.*}}:
// S390X: j .Ltmp{{.*}}
// SPARC: ta 1
// WASM: unreachable
// XTENSA: break.n
#[unsafe(no_mangle)]
pub fn do_breakpoint() {
    intrinsics::breakpoint();
}
