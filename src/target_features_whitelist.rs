use syntax::symbol::{sym, Symbol};

use rustc::session::Session;

// Copied from https://github.com/rust-lang/rust/blob/f69b07144a151f46aaee1b6230ba4160e9394562/src/librustc_codegen_llvm/llvm_util.rs#L93-L264

// WARNING: the features after applying `to_llvm_feature` must be known
// to LLVM or the feature detection code will walk past the end of the feature
// array, leading to crashes.

const ARM_WHITELIST: &[(&str, Option<Symbol>)] = &[
    ("aclass", Some(sym::arm_target_feature)),
    ("mclass", Some(sym::arm_target_feature)),
    ("rclass", Some(sym::arm_target_feature)),
    ("dsp", Some(sym::arm_target_feature)),
    ("neon", Some(sym::arm_target_feature)),
    ("v5te", Some(sym::arm_target_feature)),
    ("v6", Some(sym::arm_target_feature)),
    ("v6k", Some(sym::arm_target_feature)),
    ("v6t2", Some(sym::arm_target_feature)),
    ("v7", Some(sym::arm_target_feature)),
    ("v8", Some(sym::arm_target_feature)),
    ("vfp2", Some(sym::arm_target_feature)),
    ("vfp3", Some(sym::arm_target_feature)),
    ("vfp4", Some(sym::arm_target_feature)),
];

const AARCH64_WHITELIST: &[(&str, Option<Symbol>)] = &[
    ("fp", Some(sym::aarch64_target_feature)),
    ("neon", Some(sym::aarch64_target_feature)),
    ("sve", Some(sym::aarch64_target_feature)),
    ("crc", Some(sym::aarch64_target_feature)),
    ("crypto", Some(sym::aarch64_target_feature)),
    ("ras", Some(sym::aarch64_target_feature)),
    ("lse", Some(sym::aarch64_target_feature)),
    ("rdm", Some(sym::aarch64_target_feature)),
    ("fp16", Some(sym::aarch64_target_feature)),
    ("rcpc", Some(sym::aarch64_target_feature)),
    ("dotprod", Some(sym::aarch64_target_feature)),
    ("v8.1a", Some(sym::aarch64_target_feature)),
    ("v8.2a", Some(sym::aarch64_target_feature)),
    ("v8.3a", Some(sym::aarch64_target_feature)),
];

const X86_WHITELIST: &[(&str, Option<Symbol>)] = &[
    ("adx", Some(sym::adx_target_feature)),
    ("aes", None),
    ("avx", None),
    ("avx2", None),
    ("avx512bw", Some(sym::avx512_target_feature)),
    ("avx512cd", Some(sym::avx512_target_feature)),
    ("avx512dq", Some(sym::avx512_target_feature)),
    ("avx512er", Some(sym::avx512_target_feature)),
    ("avx512f", Some(sym::avx512_target_feature)),
    ("avx512ifma", Some(sym::avx512_target_feature)),
    ("avx512pf", Some(sym::avx512_target_feature)),
    ("avx512vbmi", Some(sym::avx512_target_feature)),
    ("avx512vl", Some(sym::avx512_target_feature)),
    ("avx512vpopcntdq", Some(sym::avx512_target_feature)),
    ("bmi1", None),
    ("bmi2", None),
    ("cmpxchg16b", Some(sym::cmpxchg16b_target_feature)),
    ("f16c", Some(sym::f16c_target_feature)),
    ("fma", None),
    ("fxsr", None),
    ("lzcnt", None),
    ("mmx", Some(sym::mmx_target_feature)),
    ("movbe", Some(sym::movbe_target_feature)),
    ("pclmulqdq", None),
    ("popcnt", None),
    ("rdrand", None),
    ("rdseed", None),
    ("rtm", Some(sym::rtm_target_feature)),
    ("sha", None),
    ("sse", None),
    ("sse2", None),
    ("sse3", None),
    ("sse4.1", None),
    ("sse4.2", None),
    ("sse4a", Some(sym::sse4a_target_feature)),
    ("ssse3", None),
    ("tbm", Some(sym::tbm_target_feature)),
    ("xsave", None),
    ("xsavec", None),
    ("xsaveopt", None),
    ("xsaves", None),
];

const HEXAGON_WHITELIST: &[(&str, Option<Symbol>)] = &[
    ("hvx", Some(sym::hexagon_target_feature)),
    ("hvx-double", Some(sym::hexagon_target_feature)),
];

const POWERPC_WHITELIST: &[(&str, Option<Symbol>)] = &[
    ("altivec", Some(sym::powerpc_target_feature)),
    ("power8-altivec", Some(sym::powerpc_target_feature)),
    ("power9-altivec", Some(sym::powerpc_target_feature)),
    ("power8-vector", Some(sym::powerpc_target_feature)),
    ("power9-vector", Some(sym::powerpc_target_feature)),
    ("vsx", Some(sym::powerpc_target_feature)),
];

const MIPS_WHITELIST: &[(&str, Option<Symbol>)] = &[
    ("fp64", Some(sym::mips_target_feature)),
    ("msa", Some(sym::mips_target_feature)),
];

const WASM_WHITELIST: &[(&str, Option<Symbol>)] = &[
    ("simd128", Some(sym::wasm_target_feature)),
    ("atomics", Some(sym::wasm_target_feature)),
];

/// When rustdoc is running, provide a list of all known features so that all their respective
/// primitives may be documented.
///
/// IMPORTANT: If you're adding another whitelist to the above lists, make sure to add it to this
/// iterator!
pub fn all_known_features() -> impl Iterator<Item=(&'static str, Option<Symbol>)> {
    ARM_WHITELIST.iter().cloned()
        .chain(AARCH64_WHITELIST.iter().cloned())
        .chain(X86_WHITELIST.iter().cloned())
        .chain(HEXAGON_WHITELIST.iter().cloned())
        .chain(POWERPC_WHITELIST.iter().cloned())
        .chain(MIPS_WHITELIST.iter().cloned())
        .chain(WASM_WHITELIST.iter().cloned())
}

pub fn target_feature_whitelist(sess: &Session)
    -> &'static [(&'static str, Option<Symbol>)]
{
    match &*sess.target.target.arch {
        "arm" => ARM_WHITELIST,
        "aarch64" => AARCH64_WHITELIST,
        "x86" | "x86_64" => X86_WHITELIST,
        "hexagon" => HEXAGON_WHITELIST,
        "mips" | "mips64" => MIPS_WHITELIST,
        "powerpc" | "powerpc64" => POWERPC_WHITELIST,
        // wasm32 on emscripten does not support these target features
        "wasm32" if !sess.target.target.options.is_like_emscripten => WASM_WHITELIST,
        _ => &[],
    }
}
