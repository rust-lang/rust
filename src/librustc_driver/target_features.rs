// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use llvm::LLVMRustHasFeature;
use rustc::session::Session;
use rustc_trans::back::write::create_target_machine;
use syntax::feature_gate::UnstableFeatures;
use syntax::symbol::Symbol;
use libc::c_char;

// WARNING: the features must be known to LLVM or the feature
// detection code will walk past the end of the feature array,
// leading to crashes.

const ARM_WHITELIST: &'static [&'static str] = &["neon\0", "vfp2\0", "vfp3\0", "vfp4\0"];

const X86_WHITELIST: &'static [&'static str] = &["avx\0", "avx2\0", "bmi\0", "bmi2\0", "sse\0",
                                                 "sse2\0", "sse3\0", "sse4.1\0", "sse4.2\0",
                                                 "ssse3\0", "tbm\0", "lzcnt\0", "popcnt\0",
                                                 "sse4a\0"];

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

    let tf = Symbol::intern("target_feature");
    for feat in whitelist {
        assert_eq!(feat.chars().last(), Some('\0'));
        if unsafe { LLVMRustHasFeature(target_machine, feat.as_ptr() as *const c_char) } {
            cfg.insert((tf, Some(Symbol::intern(&feat[..feat.len() - 1]))));
        }
    }

    let requested_features = sess.opts.cg.target_feature.split(',');
    let unstable_options = sess.opts.debugging_opts.unstable_options;
    let is_nightly = UnstableFeatures::from_environment().is_nightly_build();
    let found_negative = requested_features.clone().any(|r| r == "-crt-static");
    let found_positive = requested_features.clone().any(|r| r == "+crt-static");

    // If the target we're compiling for requests a static crt by default,
    // then see if the `-crt-static` feature was passed to disable that.
    // Otherwise if we don't have a static crt by default then see if the
    // `+crt-static` feature was passed.
    let crt_static = if sess.target.target.options.crt_static_default {
        !found_negative
    } else {
        found_positive
    };

    // If we switched from the default then that's only allowed on nightly, so
    // gate that here.
    if (found_positive || found_negative) && (!is_nightly || !unstable_options) {
        sess.fatal("specifying the `crt-static` target feature is only allowed \
                    on the nightly channel with `-Z unstable-options` passed \
                    as well");
    }

    if crt_static {
        cfg.insert((tf, Some(Symbol::intern("crt-static"))));
    }
}
