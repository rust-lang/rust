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
    "avx\0",
    "avx2\0",
    "bmi\0",
    "bmi2\0",
    "sse\0",
    "sse2\0",
    "sse3\0",
    "sse4.1\0",
    "sse4.2\0",
    "ssse3\0",
    "tbm\0",
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
