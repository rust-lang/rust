//! Runtime support needed for testing the stdarch crate.
//!
//! This basically just disassembles the current executable and then parses the
//! output once globally and then provides the `assert` function which makes
//! assertions about the disassembly of a function.
#![feature(test)] // For black_box
#![deny(rust_2018_idioms)]
#![allow(clippy::missing_docs_in_private_items, clippy::print_stdout)]

#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate cfg_if;

pub use assert_instr_macro::*;
pub use simd_test_macro::*;
use std::{cmp, collections::HashSet, env, hash, hint::black_box, str, sync::atomic::AtomicPtr};

cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
        pub mod wasm;
        use wasm::disassemble_myself;
    } else {
        mod disassembly;
        use crate::disassembly::disassemble_myself;
    }
}

lazy_static! {
    static ref DISASSEMBLY: HashSet<Function> = disassemble_myself();
}

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

    //eprintln!("shim name: {}", fnname);
    let function = &DISASSEMBLY
        .get(&Function::new(fnname))
        .unwrap_or_else(|| panic!("function \"{}\" not found in the disassembly", fnname));
    //eprintln!("  function: {:?}", function);

    let mut instrs = &function.instrs[..];
    while instrs.last().map_or(false, |s| s == "nop") {
        instrs = &instrs[..instrs.len() - 1];
    }

    // There are two cases when the expected instruction is nop:
    // 1. The expected intrinsic is compiled away so we can't
    // check for it - aka the intrinsic is not generating any code.
    // 2. It is a mark, indicating that the instruction will be
    // compiled into other instructions - mainly because of llvm
    // optimization.
    if expected == "nop" {
        return;
    }

    // Look for `expected` as the first part of any instruction in this
    // function, e.g., tzcntl in tzcntl %rax,%rax.
    let found = instrs.iter().any(|s| s.starts_with(expected));

    // Look for `call` instructions in the disassembly to detect whether
    // inlining failed: all intrinsics are `#[inline(always)]`, so
    // calling one intrinsic from another should not generate `call`
    // instructions.
    let inlining_failed = instrs.windows(2).any(|s| {
        // On 32-bit x86 position independent code will call itself and be
        // immediately followed by a `pop` to learn about the current address.
        // Let's not take that into account when considering whether a function
        // failed inlining something.
        s[0].contains("call") && (!cfg!(target_arch = "x86") || s[1].contains("pop"))
    });

    let instruction_limit = std::env::var("STDARCH_ASSERT_INSTR_LIMIT")
        .ok()
        .map_or_else(
            || match expected {
                // `cpuid` returns a pretty big aggregate structure, so exempt
                // it from the slightly more restrictive 22 instructions below.
                "cpuid" => 30,

                // Apparently, on Windows, LLVM generates a bunch of
                // saves/restores of xmm registers around these intstructions,
                // which exceeds the limit of 20 below. As it seems dictated by
                // Windows's ABI (I believe?), we probably can't do much
                // about it.
                "vzeroall" | "vzeroupper" if cfg!(windows) => 30,

                // Intrinsics using `cvtpi2ps` are typically "composites" and
                // in some cases exceed the limit.
                "cvtpi2ps" => 25,

                // core_arch/src/acle/simd32
                "usad8" => 27,
                "qadd8" | "qsub8" | "sadd8" | "sel" | "shadd8" | "shsub8" | "usub8" | "ssub8" => 29,

                // Original limit was 20 instructions, but ARM DSP Intrinsics
                // are exactly 20 instructions long. So, bump the limit to 22
                // instead of adding here a long list of exceptions.
                _ => 22,
            },
            |v| v.parse().unwrap(),
        );
    let probably_only_one_instruction = instrs.len() < instruction_limit;

    if found && probably_only_one_instruction && !inlining_failed {
        return;
    }

    // Help debug by printing out the found disassembly, and then panic as we
    // didn't find the instruction.
    println!("disassembly for {}: ", fnname,);
    for (i, instr) in instrs.iter().enumerate() {
        println!("\t{:2}: {}", i, instr);
    }

    if !found {
        panic!(
            "failed to find instruction `{}` in the disassembly",
            expected
        );
    } else if !probably_only_one_instruction {
        panic!(
            "instruction found, but the disassembly contains too many \
             instructions: #instructions = {} >= {} (limit)",
            instrs.len(),
            instruction_limit
        );
    } else if inlining_failed {
        panic!(
            "instruction found, but the disassembly contains `call` \
             instructions, which hint that inlining failed"
        );
    }
}

pub fn assert_skip_test_ok(name: &str) {
    if env::var("STDARCH_TEST_EVERYTHING").is_err() {
        return;
    }
    panic!("skipped test `{}` when it shouldn't be skipped", name);
}

// See comment in `assert-instr-macro` crate for why this exists
pub static _DONT_DEDUP: AtomicPtr<u8> = AtomicPtr::new(b"".as_ptr() as *mut _);
