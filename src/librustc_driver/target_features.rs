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
    "avx\0",                       // AVX instructions.
    "avx2\0",                      // AVX2 instructions.
    "bmi\0",                       // BMI instructions.
    "bmi2\0",                      // BMI2 instructions.
    "sse\0",                       // SSE instructions.
    "sse-unaligned-mem\0",         // Allow unaligned memory operands with SSE instructions.
    "sse2\0",                      // SSE2 instructions.
    "sse3\0",                      // SSE3 instructions.
    "sse4.1\0",                    // SSE 4.1 instructions.
    "sse4.2\0",                    // SSE 4.2 instructions.
    "sse4a\0",                     // SSE 4a instructions.
    "ssse3\0",                     // SSSE3 instructions.
    "tbm\0",                       // TBM instructions.
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
