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
use rustc::session::Session;
use syntax::parse::token::InternedString;
use syntax::parse::token::intern_and_get_ident as intern;

/// Add `target_feature = "..."` cfgs for a variety of platform
/// specific features (SSE, NEON etc.).
///
/// This uses a scheme similar to that employed by clang: reimplement
/// the target feature knowledge. *Theoretically* we could query LLVM
/// since that has perfect knowledge about what things are enabled in
/// code-generation, however, it is extremely non-obvious how to do
/// this successfully. Each platform defines a subclass of a
/// SubtargetInfo, which knows all this information, but the ways to
/// query them do not seem to be public.
pub fn add_configuration(cfg: &mut ast::CrateConfig, sess: &Session) {
    let tf = InternedString::new("target_feature");
    macro_rules! fillout {
        ($($func: ident, $name: expr;)*) => {{
            $(if $func(sess) {
                cfg.push(attr::mk_name_value_item_str(tf.clone(), intern($name)))
            })*
        }}
    }
    fillout! {
        has_sse, "sse";
        has_sse2, "sse2";
        has_sse3, "sse3";
        has_ssse3, "ssse3";
        has_sse41, "sse4.1";
        has_sse42, "sse4.2";
        has_avx, "avx";
        has_avx2, "avx2";
        has_neon, "neon";
        has_vfp, "vfp";
    }
}


fn features_contain(sess: &Session, s: &str) -> bool {
    sess.target.target.options.features.contains(s) || sess.opts.cg.target_feature.contains(s)
}

pub fn has_sse(sess: &Session) -> bool {
    features_contain(sess, "+sse") || has_sse2(sess)
}
pub fn has_sse2(sess: &Session) -> bool {
    // x86-64 requires at least SSE2 support
    sess.target.target.arch == "x86_64" || features_contain(sess, "+sse2") || has_sse3(sess)
}
pub fn has_sse3(sess: &Session) -> bool {
    features_contain(sess, "+sse3") || has_ssse3(sess)
}
pub fn has_ssse3(sess: &Session) -> bool {
    features_contain(sess, "+ssse3") || has_sse41(sess)
}
pub fn has_sse41(sess: &Session) -> bool {
    features_contain(sess, "+sse4.1") || has_sse42(sess)
}
pub fn has_sse42(sess: &Session) -> bool {
    features_contain(sess, "+sse4.2") || has_avx(sess)
}
pub fn has_avx(sess: &Session) -> bool {
    features_contain(sess, "+avx") || has_avx2(sess)
}
pub fn has_avx2(sess: &Session) -> bool {
    features_contain(sess, "+avx2")
}

pub fn has_neon(sess: &Session) -> bool {
    // AArch64 requires NEON support
    sess.target.target.arch == "aarch64" || features_contain(sess, "+neon")
}
pub fn has_vfp(sess: &Session) -> bool {
    // AArch64 requires VFP support
    sess.target.target.arch == "aarch64" || features_contain(sess, "+vfp")
}
