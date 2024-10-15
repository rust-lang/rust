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

    let disassembly = llvm_objdump().disassemble().input("compiled").run().stdout_utf8();

    // search for the following instruction sequence:
    // ```disassembly
    // 00000080 <main>:
    // 80: 81 e0         ldi     r24, 0x1
    // 82: 92 e0         ldi     r25, 0x2
    // 84: 85 b9         out     0x5, r24
    // 86: 95 b9         out     0x5, r25
    // 88: fd cf         rjmp    .-6
    // ```
    // This matches on all instructions, since the size of the instructions be-
    // fore the relative jump has an impact on the label offset. Old versions
    // of the Rust compiler did produce a label `rjmp .-4` (misses the first
    // instruction in the loop).
    disassembly
        .trim()
        .lines()
        .skip_while(|&line| !line.contains("<main>"))
        .inspect(|line| println!("{line}"))
        .skip(1)
        .zip(["ldi\t", "ldi\t", "out\t", "out\t", "rjmp\t.-6"])
        .for_each(|(line, expected_instruction)| {
            assert!(
                line.contains(expected_instruction),
                "expected instruction `{expected_instruction}`, got `{line}`"
            );
        });
}
