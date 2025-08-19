//@ needs-llvm-components: avr
//@ needs-rust-lld
//! Regression test for #129301/llvm-project#106722 within `rustc`.
//!
//! Some LLVM-versions had wrong offsets in the local labels, causing the first
//! loop instruction to be missed. This test therefore contains a simple loop
//! with trivial instructions in it, to see, where the label is placed.
//!
//! This must be a `rmake`-test and cannot be a `tests/assembly-llvm/`-test, since the
//! wrong output is only produced with direct assembly generation, but not when
//! "emit-asm" is used, as described in the issue description of #129301:
//! https://github.com/rust-lang/rust/issues/129301#issue-2475070770

// FIXME(#133480): this has been randomly failing on `x86_64-mingw` due to linker hangs or
// crashes... so I'm going to disable this test for windows for now.
//@ ignore-windows-gnu

use run_make_support::{llvm_objdump, rustc};

fn main() {
    rustc()
        .input("avr-rjmp-offsets.rs")
        .opt_level("s")
        .panic("abort")
        .target("avr-none")
        // rust-lld has some troubles understanding the -mmcu flag, so for the
        // time being let's tell rustc to emit binary that's compatible with the
        // target CPU that lld defaults to, i.e. just `avr` (that's simply the
        // minimal common instruction set across all AVRs)
        .target_cpu("avr")
        // normally one links with `avr-gcc`, but this is not available in CI,
        // hence this test diverges from the default behavior to enable linking
        // at all, which is necessary for the test (to resolve the labels). To
        // not depend on a special linker script, the main-function is marked as
        // the entry function, causing the linker to not remove it.
        .linker("rust-lld")
        .link_arg("--entry=main")
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
    assert!(disassembly.contains("<main>"), "no main function in output");

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
