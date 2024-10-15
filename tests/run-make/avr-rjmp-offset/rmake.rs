//@ needs-llvm-components: avr
//! Regression test for #129301/llvm-project#106722 within `rustc`.
//!
//! Some LLVM-versions had wrong offsets in the local labels, causing the first
//! loop instruction to be missed. This test therefore contains a simple loop
//! with trivial instructions in it, to see, where the label is placed.
//!
//! This must be a `rmake`-test and cannot be a `tests/assembly`-test, since the
//! wrong output is only produced with direct assembly generation, but not when
//! "emit-asm" is used, as described in the issue description of #129301:
//! https://github.com/rust-lang/rust/issues/129301#issue-2475070770
use run_make_support::{llvm_objdump, rustc};

fn main() {
    rustc()
        .input("avr-rjmp-offsets.rs")
        .opt_level("s")
        .panic("abort")
        .target("avr-unknown-gnu-atmega328")
        .output("compiled")
        .run();

    llvm_objdump()
        .disassemble()
        .input("compiled")
        .run()
        .assert_stdout_contains_regex(r"rjmp.*\.-14");
}
