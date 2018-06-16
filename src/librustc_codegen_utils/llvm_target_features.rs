// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::session::Session;

// WARNING: the features after applying `to_llvm_feature` must be known
// to LLVM or the feature detection code will walk past the end of the feature
// array, leading to crashes.

const ARM_WHITELIST: &[(&str, Option<&str>)] = &[
    ("mclass", Some("arm_target_feature")),
    ("neon", Some("arm_target_feature")),
    ("v7", Some("arm_target_feature")),
    ("vfp2", Some("arm_target_feature")),
    ("vfp3", Some("arm_target_feature")),
    ("vfp4", Some("arm_target_feature")),
];

const AARCH64_WHITELIST: &[(&str, Option<&str>)] = &[
    ("fp", Some("aarch64_target_feature")),
    ("neon", Some("aarch64_target_feature")),
    ("sve", Some("aarch64_target_feature")),
    ("crc", Some("aarch64_target_feature")),
    ("crypto", Some("aarch64_target_feature")),
    ("ras", Some("aarch64_target_feature")),
    ("lse", Some("aarch64_target_feature")),
    ("rdm", Some("aarch64_target_feature")),
    ("fp16", Some("aarch64_target_feature")),
    ("rcpc", Some("aarch64_target_feature")),
    ("dotprod", Some("aarch64_target_feature")),
    ("v8.1a", Some("aarch64_target_feature")),
    ("v8.2a", Some("aarch64_target_feature")),
    ("v8.3a", Some("aarch64_target_feature")),
];

const X86_WHITELIST: &[(&str, Option<&str>)] = &[
    ("aes", None),
    ("avx", None),
    ("avx2", None),
    ("avx512bw", Some("avx512_target_feature")),
    ("avx512cd", Some("avx512_target_feature")),
    ("avx512dq", Some("avx512_target_feature")),
    ("avx512er", Some("avx512_target_feature")),
    ("avx512f", Some("avx512_target_feature")),
    ("avx512ifma", Some("avx512_target_feature")),
    ("avx512pf", Some("avx512_target_feature")),
    ("avx512vbmi", Some("avx512_target_feature")),
    ("avx512vl", Some("avx512_target_feature")),
    ("avx512vpopcntdq", Some("avx512_target_feature")),
    ("bmi1", None),
    ("bmi2", None),
    ("fma", None),
    ("fxsr", None),
    ("lzcnt", None),
    ("mmx", Some("mmx_target_feature")),
    ("pclmulqdq", None),
    ("popcnt", None),
    ("rdrand", None),
    ("rdseed", None),
    ("sha", None),
    ("sse", None),
    ("sse2", None),
    ("sse3", None),
    ("sse4.1", None),
    ("sse4.2", None),
    ("sse4a", Some("sse4a_target_feature")),
    ("ssse3", None),
    ("tbm", Some("tbm_target_feature")),
    ("xsave", None),
    ("xsavec", None),
    ("xsaveopt", None),
    ("xsaves", None),
];

const HEXAGON_WHITELIST: &[(&str, Option<&str>)] = &[
    ("hvx", Some("hexagon_target_feature")),
    ("hvx-double", Some("hexagon_target_feature")),
];

const POWERPC_WHITELIST: &[(&str, Option<&str>)] = &[
    ("altivec", Some("powerpc_target_feature")),
    ("power8-altivec", Some("powerpc_target_feature")),
    ("power9-altivec", Some("powerpc_target_feature")),
    ("power8-vector", Some("powerpc_target_feature")),
    ("power9-vector", Some("powerpc_target_feature")),
    ("vsx", Some("powerpc_target_feature")),
];

const MIPS_WHITELIST: &[(&str, Option<&str>)] = &[
    ("fp64", Some("mips_target_feature")),
    ("msa", Some("mips_target_feature")),
];

/// When rustdoc is running, provide a list of all known features so that all their respective
/// primtives may be documented.
///
/// IMPORTANT: If you're adding another whitelist to the above lists, make sure to add it to this
/// iterator!
pub fn all_known_features() -> impl Iterator<Item=(&'static str, Option<&'static str>)> {
    ARM_WHITELIST.iter().cloned()
        .chain(AARCH64_WHITELIST.iter().cloned())
        .chain(X86_WHITELIST.iter().cloned())
        .chain(HEXAGON_WHITELIST.iter().cloned())
        .chain(POWERPC_WHITELIST.iter().cloned())
        .chain(MIPS_WHITELIST.iter().cloned())
}

pub fn target_feature_whitelist(sess: &Session)
    -> &'static [(&'static str, Option<&'static str>)]
{
    match &*sess.target.target.arch {
        "arm" => ARM_WHITELIST,
        "aarch64" => AARCH64_WHITELIST,
        "x86" | "x86_64" => X86_WHITELIST,
        "hexagon" => HEXAGON_WHITELIST,
        "mips" | "mips64" => MIPS_WHITELIST,
        "powerpc" | "powerpc64" => POWERPC_WHITELIST,
        _ => &[],
    }
}
