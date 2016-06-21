// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::{ast, attr};
use llvm::LLVMRustHasFeature;
use rustc::session::Session;
use rustc_trans::back::write::create_target_machine;
use syntax::parse::token::InternedString;
use syntax::parse::token::intern_and_get_ident as intern;
use libc::c_char;

// WARNING: the features must be known to LLVM or the feature
// detection code will walk past the end of the feature array,
// leading to crashes.

const ARM_WHITELIST: &'static [&'static str] = &[
    "neon\0",
    "vfp2\0",
    "vfp3\0",
    "vfp4\0",
];

const X86_WHITELIST: &'static [&'static str] = &[
    "16bit-mode\0",                // 16-bit mode (i8086).
    "32bit-mode\0",                // 32-bit mode (80386).
    "3dnow\0",                     // 3DNow! instructions.
    "3dnowa\0",                    // 3DNow! Athlon instructions.
    "64bit\0",                     // Support 64-bit instructions.
    "64bit-mode\0",                // 64-bit mode (x86_64).
    "adx\0",                       // Support ADX instructions.
    "aes\0",                       // AES instructions.
    "atom\0",                      // Intel Atom processors.
    "avx\0",                       // AVX instructions.
    "avx2\0",                      // AVX2 instructions.
    "avx512bw\0",                  // AVX-512 Byte and Word Instructions.
    "avx512cd\0",                  // AVX-512 Conflict Detection Instructions.
    "avx512dq\0",                  // AVX-512 Doubleword and Quadword Instructions.
    "avx512er\0",                  // AVX-512 Exponential and Reciprocal Instructions.
    "avx512f\0",                   // AVX-512 instructions.
    "avx512ifma\0",                // AVX-512 Integer Fused Multiple-Add.
    "avx512pf\0",                  // AVX-512 PreFetch Instructions.
    "avx512vbmi\0",                // AVX-512 Vector Bit Manipulation Instructions.
    "avx512vl\0",                  // AVX-512 Vector Length eXtensions.
    "bmi\0",                       // BMI instructions.
    "bmi2\0",                      // BMI2 instructions.
    "call-reg-indirect\0",         // Call register indirect.
    "clflushopt\0",                // Flush A Cache Line Optimized.
    "clwb\0",                      // Cache Line Write Back.
    "cmov\0",                      // Conditional move instructions.
    "cx16\0",                      // 64-bit with cmpxchg16b.
    "f16c\0",                      // 16-bit floating point conversion instructions.
    "fast-partial-ymm-write\0",    // Partial writes to YMM registers are fast.
    "fma\0",                       // Three-operand fused multiple-add.
    "fma4\0",                      // Four-operand fused multiple-add.
    "fsgsbase\0",                  // FS/GS Base instructions.
    "fxsr\0",                      // fxsave/fxrestore instructions.
    "hle\0",                       // HLE.
    "idivl-to-divb\0",             // Use 8-bit divide for positive values less than 256.
    "idivq-to-divw\0",             // Use 16-bit divide for positive values less than 65536.
    "invpcid\0",                   // Invalidate Process-Context Identifier.
    "lea-sp\0",                    // Uses LEA for adjusting the stack pointer.
    "lea-uses-ag\0",               // LEA instruction needs inputs at AG stage.
    "lzcnt\0",                     // LZCNT instruction.
    "mmx\0",                       // MMX instructions.
    "movbe\0",                     // MOVBE instruction.
    "mpx\0",                       // MPX instructions.
    "mwaitx\0",                    // MONITORX/MWAITX timer functionality.
    "pad-short-functions\0",       // Pad short functions.
    "pclmul\0",                    // Packed carry-less multiplication instructions.
    "pcommit\0",                   // Persistent Commit.
    "pku\0",                       // Protection keys.
    "popcnt\0",                    // POPCNT instruction.
    "prefetchwt1\0",               // Prefetch with Intent to Write and T1 Hint.
    "prfchw\0",                    // PRFCHW instructions.
    "rdrnd\0",                     // RDRAND instruction.
    "rdseed\0",                    // RDSEED instruction.
    "rtm\0",                       // RTM instructions.
    "sahf\0",                      // LAHF and SAHF instructions.
    "sgx\0",                       // Software Guard Extensions.
    "sha\0",                       // SHA instructions.
    "slm\0",                       // Intel Silvermont processors.
    "slow-bt-mem\0",               // Bit testing of memory is slow.
    "slow-incdec\0",               // INC and DEC instructions are slower than ADD and SUB.
    "slow-lea\0",                  // LEA instruction with certain arguments is slow.
    "slow-shld\0",                 // SHLD instruction is slow.
    "slow-unaligned-mem-16\0",     // Slow unaligned 16-byte memory access.
    "slow-unaligned-mem-32\0",     // Slow unaligned 32-byte memory access.
    "smap\0",                      // Supervisor Mode Access Protection.
    "soft-float\0",                // Use software floating point features.
    "sse\0",                       // SSE instructions.
    "sse-unaligned-mem\0",         // Allow unaligned memory operands with SSE instructions.
    "sse2\0",                      // SSE2 instructions.
    "sse3\0",                      // SSE3 instructions.
    "sse4.1\0",                    // SSE 4.1 instructions.
    "sse4.2\0",                    // SSE 4.2 instructions.
    "sse4a\0",                     // SSE 4a instructions.
    "ssse3\0",                     // SSSE3 instructions.
    "tbm\0",                       // TBM instructions.
    "vmfunc\0",                    // VM Functions.
    "x87\0",                       // X87 float instructions.
    "xop\0",                       // XOP instructions.
    "xsave\0",                     // xsave instructions.
    "xsavec\0",                    // xsavec instructions.
    "xsaveopt\0",                  // xsaveopt instructions.
    "xsaves\0"                     // xsaves instructions.
];

/// Add `target_feature = "..."` cfgs for a variety of platform
/// specific features (SSE, NEON etc.).
///
/// This is performed by checking whether a whitelisted set of
/// features is available on the target machine, by querying LLVM.
pub fn add_configuration(cfg: &mut ast::CrateConfig, sess: &Session) {
    let target_machine = create_target_machine(sess);

    let whitelist = match &*sess.target.target.arch {
        "arm" => ARM_WHITELIST,
        "x86" | "x86_64" => X86_WHITELIST,
        _ => &[],
    };

    let tf = InternedString::new("target_feature");
    for feat in whitelist {
        assert_eq!(feat.chars().last(), Some('\0'));
        if unsafe { LLVMRustHasFeature(target_machine, feat.as_ptr() as *const c_char) } {
            cfg.push(attr::mk_name_value_item_str(tf.clone(), intern(&feat[..feat.len()-1])))
        }
    }
}
