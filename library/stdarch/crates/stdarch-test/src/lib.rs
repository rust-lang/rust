//! Runtime support needed for testing the stdarch crate.
//!
//! This basically just disassembles the current executable and then parses the
//! output once globally and then provides the `assert` function which makes
//! assertions about the disassembly of a function.
#![deny(rust_2018_idioms)]
#![allow(clippy::missing_docs_in_private_items, clippy::print_stdout)]

#[macro_use]
extern crate cfg_if;

pub use assert_instr_macro::*;
pub use simd_test_macro::*;
use std::{cmp, collections::HashSet, env, hash, hint::black_box, str, sync::LazyLock};

cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
        pub mod wasm;
        use wasm::disassemble_myself;
    } else {
        mod disassembly;
        use crate::disassembly::disassemble_myself;
    }
}

static DISASSEMBLY: LazyLock<HashSet<Function>> = LazyLock::new(disassemble_myself);

#[derive(Debug)]
struct Function {
    name: String,
    instrs: Vec<String>,
}
impl Function {
    fn new(n: &str) -> Self {
        Self {
            name: n.to_string(),
            instrs: Vec::new(),
        }
    }
}

impl cmp::PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl cmp::Eq for Function {}

impl hash::Hash for Function {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

/// Main entry point for this crate, called by the `#[assert_instr]` macro.
///
/// This asserts that the function at `fnptr` contains the instruction
/// `expected` provided.
pub fn assert(shim_addr: usize, fnname: &str, expected: &str) {
    // Make sure that the shim is not removed
    black_box(shim_addr);

    //eprintln!("shim name: {fnname}");
    let Some(function) = &DISASSEMBLY.get(&Function::new(fnname)) else {
        panic!("function `{fnname}` not found in the disassembly")
    };
    //eprintln!("  function: {:?}", function);

    // Trim any filler instructions.
    let mut instrs = &function.instrs[..];
    while instrs.last().is_some_and(|s| s == "nop" || s == "int3") {
        instrs = &instrs[..instrs.len() - 1];
    }

    // Look for `expected` as the first part of any instruction in this
    // function, e.g., tzcntl in tzcntl %rax,%rax.
    //
    // There are two cases when the expected instruction is nop:
    // 1. The expected intrinsic is compiled away so we can't
    // check for it - aka the intrinsic is not generating any code.
    // 2. It is a mark, indicating that the instruction will be
    // compiled into other instructions - mainly because of llvm
    // optimization.
    let expected = match expected {
        // `<unknown>` is what LLVM will generate for unknown instructions. We use this to fail
        // loudly when LLVM does start supporting these instructions.
        //
        // This was introduced in https://github.com/rust-lang/stdarch/pull/1674 to work around the
        // RISC-V P extension not yet being supported.
        "unknown" => "<unknown>",
        _ => expected,
    };

    // Check whether the given instruction is part of the disassemblied body.
    let found = expected == "nop"
        || instrs.iter().any(|instruction| {
            instruction.starts_with(expected)
            // Check that the next character is non-alphanumeric. This prevents false negatives
            // when e.g. `fminnm` was used but `fmin` was expected.
            //
            // TODO: resolve the conflicts (x86_64 and aarch64 have a bunch, probably others)
            // && !instruction[expected.len()..].starts_with(|c: char| c.is_ascii_alphanumeric())
        });

    // Look for subroutine call instructions in the disassembly to detect whether
    // inlining failed: all intrinsics are `#[inline(always)]`, so calling one
    // intrinsic from another should not generate subroutine call instructions.
    let inlining_failed = if cfg!(target_arch = "x86_64") || cfg!(target_arch = "wasm32") {
        instrs.iter().any(|s| s.starts_with("call "))
    } else if cfg!(target_arch = "x86") {
        instrs.windows(2).any(|s| {
            // On 32-bit x86 position independent code will call itself and be
            // immediately followed by a `pop` to learn about the current address.
            // Let's not take that into account when considering whether a function
            // failed inlining something.
            s[0].starts_with("call ") && s[1].starts_with("pop") // FIXME: original logic but does not match comment
        })
    } else if cfg!(any(
        target_arch = "aarch64",
        target_arch = "arm64ec",
        target_arch = "powerpc",
        target_arch = "powerpc64"
    )) {
        instrs.iter().any(|s| s.starts_with("bl "))
    } else {
        // FIXME: Add detection for other archs
        false
    };

    let instruction_limit = std::env::var("STDARCH_ASSERT_INSTR_LIMIT")
        .ok()
        .map_or_else(
            || match expected {
                // `cpuid` returns a pretty big aggregate structure, so exempt
                // it from the slightly more restrictive 22 instructions below.
                "cpuid" => 30,

                // These require 8 loads and stores, so it _just_ overflows the limit
                "aesencwide128kl" | "aesencwide256kl" | "aesdecwide128kl" | "aesdecwide256kl" => 24,

                // Apparently, on Windows, LLVM generates a bunch of
                // saves/restores of xmm registers around these instructions,
                // which exceeds the limit of 20 below. As it seems dictated by
                // Windows's ABI (I believe?), we probably can't do much
                // about it.
                "vzeroall" | "vzeroupper" if cfg!(windows) => 30,

                // Intrinsics using `cvtpi2ps` are typically "composites" and
                // in some cases exceed the limit.
                "cvtpi2ps" => 25,
                // core_arch/src/arm_shared/simd32
                // vfmaq_n_f32_vfma : #instructions = 26 >= 22 (limit)
                "usad8" | "vfma" | "vfms" => 27,
                "qadd8" | "qsub8" | "sadd8" | "sel" | "shadd8" | "shsub8" | "usub8" | "ssub8" => 29,
                // core_arch/src/arm_shared/simd32
                // vst1q_s64_x4_vst1 : #instructions = 27 >= 22 (limit)
                "vld3" => 28,
                // core_arch/src/arm_shared/simd32
                // vld4q_lane_u32_vld4 : #instructions = 36 >= 22 (limit)
                "vld4" => 37,
                // core_arch/src/arm_shared/simd32
                // vst1q_s64_x4_vst1 : #instructions = 40 >= 22 (limit)
                "vst1" => 41,
                // core_arch/src/arm_shared/simd32
                // vst3q_u32_vst3 : #instructions = 25 >= 22 (limit)
                "vst3" => 26,
                // core_arch/src/arm_shared/simd32
                // vst4q_u32_vst4 : #instructions = 33 >= 22 (limit)
                "vst4" => 34,

                // core_arch/src/arm_shared/simd32
                // vst1q_p64_x4_nop : #instructions = 33 >= 22 (limit)
                "nop" if fnname.contains("vst1q_p64") => 34,

                // Original limit was 20 instructions, but ARM DSP Intrinsics
                // are exactly 20 instructions long. So, bump the limit to 22
                // instead of adding here a long list of exceptions.
                _ => {
                    // aarch64_be may add reverse instructions which increases
                    // the number of instructions generated.
                    if cfg!(all(target_endian = "big", target_arch = "aarch64")) {
                        32
                    } else {
                        22
                    }
                }
            },
            |v| v.parse().unwrap(),
        );
    let probably_only_one_instruction = instrs.len() < instruction_limit;

    if found && probably_only_one_instruction && !inlining_failed {
        return;
    }

    // Help debug by printing out the found disassembly, and then panic as we
    // didn't find the instruction.
    println!("disassembly for {fnname}: ",);
    for (i, instr) in instrs.iter().enumerate() {
        println!("\t{i:2}: {instr}");
    }

    if !found {
        panic!("failed to find instruction `{expected}` in the disassembly");
    } else if !probably_only_one_instruction {
        panic!(
            "instruction found, but the disassembly contains too many \
             instructions: #instructions = {} >= {} (limit)",
            instrs.len(),
            instruction_limit
        );
    } else if inlining_failed {
        panic!(
            "instruction found, but the disassembly contains subroutine \
             call instructions, which hint that inlining failed"
        );
    }
}

pub fn assert_skip_test_ok(name: &str, missing_features: &[&str]) {
    println!("Skipping test `{name}` due to missing target features:");
    for feature in missing_features {
        println!("  - {feature}");
    }
    match env::var("STDARCH_TEST_EVERYTHING") {
        Ok(_) => panic!("skipped test `{name}` when it shouldn't be skipped"),
        Err(_) => println!("Set STDARCH_TEST_EVERYTHING to make this an error."),
    }
}
