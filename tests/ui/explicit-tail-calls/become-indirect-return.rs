//@ run-pass
//@ ignore-backends: gcc
//@ ignore-wasm 'tail-call' feature not enabled in target wasm32-wasip1
//@ ignore-cross-compile
//
// LLVM musttail support is incomplete for these targets.
// See https://github.com/rust-lang/rust/issues/148748 for the target test matrix.
// See https://github.com/llvm/llvm-project/issues/63214 for AIX and PowerPC.
// See https://github.com/llvm/llvm-project/issues/57795 for MIPS and MIPS64.
//@ ignore-aix
//@ ignore-csky
//@ ignore-mips
//@ ignore-mips64
//@ ignore-powerpc
//@ ignore-powerpc64
//
// LLVM musttail support is lacking for sret lowering on these targets.
// Returning `[u8; 24]` uses sret lowering here.
// See https://github.com/llvm/llvm-project/issues/157814 for RISC-V.
// See https://github.com/llvm/llvm-project/issues/168152 for LoongArch.
// RISC-V fix: https://github.com/llvm/llvm-project/pull/185094, should be in LLVM 23.
// LoongArch will likely be fixed with similar changes.
//@ ignore-riscv64
//@ ignore-loongarch32
//@ ignore-loongarch64
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

#[inline(never)]
fn op_dummy(_param: &Box<u8>) -> [u8; 24] {
    [1; 24]
}

#[inline(never)]
fn dispatch(param: &Box<u8>) -> [u8; 24] {
    become op_dummy(param)
}

fn main() {
    let param = Box::new(0);
    let result = dispatch(&param);
    assert_eq!(result, [1; 24]);
}
