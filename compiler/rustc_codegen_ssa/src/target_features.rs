use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::ty::query::Providers;
use rustc_session::Session;
use rustc_span::symbol::sym;
use rustc_span::symbol::Symbol;

const ARM_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
    ("aclass", Some(sym::arm_target_feature)),
    ("mclass", Some(sym::arm_target_feature)),
    ("rclass", Some(sym::arm_target_feature)),
    ("dsp", Some(sym::arm_target_feature)),
    ("neon", Some(sym::arm_target_feature)),
    ("crc", Some(sym::arm_target_feature)),
    ("crypto", Some(sym::arm_target_feature)),
    ("v5te", Some(sym::arm_target_feature)),
    ("v6", Some(sym::arm_target_feature)),
    ("v6k", Some(sym::arm_target_feature)),
    ("v6t2", Some(sym::arm_target_feature)),
    ("v7", Some(sym::arm_target_feature)),
    ("v8", Some(sym::arm_target_feature)),
    ("vfp2", Some(sym::arm_target_feature)),
    ("vfp3", Some(sym::arm_target_feature)),
    ("vfp4", Some(sym::arm_target_feature)),
    // This is needed for inline assembly, but shouldn't be stabilized as-is
    // since it should be enabled per-function using #[instruction_set], not
    // #[target_feature].
    ("thumb-mode", Some(sym::arm_target_feature)),
];

const AARCH64_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
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
    ("tme", Some(sym::aarch64_target_feature)),
    ("v8.1a", Some(sym::aarch64_target_feature)),
    ("v8.2a", Some(sym::aarch64_target_feature)),
    ("v8.3a", Some(sym::aarch64_target_feature)),
];

const X86_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
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
    ("ermsb", Some(sym::ermsb_target_feature)),
    ("f16c", Some(sym::f16c_target_feature)),
    ("fma", None),
    ("fxsr", None),
    ("lzcnt", None),
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

const HEXAGON_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
    ("hvx", Some(sym::hexagon_target_feature)),
    ("hvx-length128b", Some(sym::hexagon_target_feature)),
];

const POWERPC_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
    ("altivec", Some(sym::powerpc_target_feature)),
    ("power8-altivec", Some(sym::powerpc_target_feature)),
    ("power9-altivec", Some(sym::powerpc_target_feature)),
    ("power8-vector", Some(sym::powerpc_target_feature)),
    ("power9-vector", Some(sym::powerpc_target_feature)),
    ("vsx", Some(sym::powerpc_target_feature)),
];

const MIPS_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] =
    &[("fp64", Some(sym::mips_target_feature)), ("msa", Some(sym::mips_target_feature))];

const RISCV_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
    ("m", Some(sym::riscv_target_feature)),
    ("a", Some(sym::riscv_target_feature)),
    ("c", Some(sym::riscv_target_feature)),
    ("f", Some(sym::riscv_target_feature)),
    ("d", Some(sym::riscv_target_feature)),
    ("e", Some(sym::riscv_target_feature)),
];

const WASM_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
    ("simd128", Some(sym::wasm_target_feature)),
    ("atomics", Some(sym::wasm_target_feature)),
    ("nontrapping-fptoint", Some(sym::wasm_target_feature)),
];

/// When rustdoc is running, provide a list of all known features so that all their respective
/// primitives may be documented.
///
/// IMPORTANT: If you're adding another feature list above, make sure to add it to this iterator!
pub fn all_known_features() -> impl Iterator<Item = (&'static str, Option<Symbol>)> {
    std::iter::empty()
        .chain(ARM_ALLOWED_FEATURES.iter())
        .chain(AARCH64_ALLOWED_FEATURES.iter())
        .chain(X86_ALLOWED_FEATURES.iter())
        .chain(HEXAGON_ALLOWED_FEATURES.iter())
        .chain(POWERPC_ALLOWED_FEATURES.iter())
        .chain(MIPS_ALLOWED_FEATURES.iter())
        .chain(RISCV_ALLOWED_FEATURES.iter())
        .chain(WASM_ALLOWED_FEATURES.iter())
        .cloned()
}

pub fn supported_target_features(sess: &Session) -> &'static [(&'static str, Option<Symbol>)] {
    match &*sess.target.arch {
        "arm" => ARM_ALLOWED_FEATURES,
        "aarch64" => AARCH64_ALLOWED_FEATURES,
        "x86" | "x86_64" => X86_ALLOWED_FEATURES,
        "hexagon" => HEXAGON_ALLOWED_FEATURES,
        "mips" | "mips64" => MIPS_ALLOWED_FEATURES,
        "powerpc" | "powerpc64" => POWERPC_ALLOWED_FEATURES,
        "riscv32" | "riscv64" => RISCV_ALLOWED_FEATURES,
        "wasm32" => WASM_ALLOWED_FEATURES,
        _ => &[],
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.supported_target_features = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        if tcx.sess.opts.actually_rustdoc {
            // rustdoc needs to be able to document functions that use all the features, so
            // whitelist them all
            all_known_features().map(|(a, b)| (a.to_string(), b)).collect()
        } else {
            supported_target_features(tcx.sess).iter().map(|&(a, b)| (a.to_string(), b)).collect()
        }
    };
}
