use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::ty::query::Providers;
use rustc_session::Session;
use rustc_span::symbol::sym;
use rustc_span::symbol::Symbol;

/// Features that control behaviour of rustc, rather than the codegen.
pub const RUSTC_SPECIFIC_FEATURES: &[&str] = &["crt-static"];

// When adding features to the below lists
// check whether they're named already elsewhere in rust
// e.g. in stdarch and whether the given name matches LLVM's
// if it doesn't, to_llvm_feature in llvm_util in rustc_codegen_llvm needs to be adapted

const ARM_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
    ("aclass", Some(sym::arm_target_feature)),
    ("mclass", Some(sym::arm_target_feature)),
    ("rclass", Some(sym::arm_target_feature)),
    ("dsp", Some(sym::arm_target_feature)),
    ("neon", Some(sym::arm_target_feature)),
    ("crc", Some(sym::arm_target_feature)),
    ("crypto", Some(sym::arm_target_feature)),
    ("aes", Some(sym::arm_target_feature)),
    ("sha2", Some(sym::arm_target_feature)),
    ("i8mm", Some(sym::arm_target_feature)),
    ("dotprod", Some(sym::arm_target_feature)),
    ("v5te", Some(sym::arm_target_feature)),
    ("v6", Some(sym::arm_target_feature)),
    ("v6k", Some(sym::arm_target_feature)),
    ("v6t2", Some(sym::arm_target_feature)),
    ("v7", Some(sym::arm_target_feature)),
    ("v8", Some(sym::arm_target_feature)),
    ("vfp2", Some(sym::arm_target_feature)),
    ("vfp3", Some(sym::arm_target_feature)),
    ("vfp4", Some(sym::arm_target_feature)),
    ("fp-armv8", Some(sym::arm_target_feature)),
    // This is needed for inline assembly, but shouldn't be stabilized as-is
    // since it should be enabled per-function using #[instruction_set], not
    // #[target_feature].
    ("thumb-mode", Some(sym::arm_target_feature)),
    ("thumb2", Some(sym::arm_target_feature)),
    ("d32", Some(sym::arm_target_feature)),
];

const AARCH64_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
    // FEAT_AdvSimd & FEAT_FP
    ("neon", None),
    // FEAT_FP16
    ("fp16", None),
    // FEAT_SVE
    ("sve", None),
    // FEAT_CRC
    ("crc", None),
    // FEAT_RAS
    ("ras", None),
    // FEAT_LSE
    ("lse", None),
    // FEAT_RDM
    ("rdm", None),
    // FEAT_RCPC
    ("rcpc", None),
    // FEAT_RCPC2
    ("rcpc2", None),
    // FEAT_DotProd
    ("dotprod", None),
    // FEAT_TME
    ("tme", None),
    // FEAT_FHM
    ("fhm", None),
    // FEAT_DIT
    ("dit", None),
    // FEAT_FLAGM
    ("flagm", None),
    // FEAT_SSBS
    ("ssbs", None),
    // FEAT_SB
    ("sb", None),
    // FEAT_PAUTH (address authentication)
    ("paca", None),
    // FEAT_PAUTH (generic authentication)
    ("pacg", None),
    // FEAT_DPB
    ("dpb", None),
    // FEAT_DPB2
    ("dpb2", None),
    // FEAT_SVE2
    ("sve2", None),
    // FEAT_SVE2_AES
    ("sve2-aes", None),
    // FEAT_SVE2_SM4
    ("sve2-sm4", None),
    // FEAT_SVE2_SHA3
    ("sve2-sha3", None),
    // FEAT_SVE2_BitPerm
    ("sve2-bitperm", None),
    // FEAT_FRINTTS
    ("frintts", None),
    // FEAT_I8MM
    ("i8mm", None),
    // FEAT_F32MM
    ("f32mm", None),
    // FEAT_F64MM
    ("f64mm", None),
    // FEAT_BF16
    ("bf16", None),
    // FEAT_RAND
    ("rand", None),
    // FEAT_BTI
    ("bti", None),
    // FEAT_MTE
    ("mte", None),
    // FEAT_JSCVT
    ("jsconv", None),
    // FEAT_FCMA
    ("fcma", None),
    // FEAT_AES
    ("aes", None),
    // FEAT_SHA1 & FEAT_SHA256
    ("sha2", None),
    // FEAT_SHA512 & FEAT_SHA3
    ("sha3", None),
    // FEAT_SM3 & FEAT_SM4
    ("sm4", None),
    // FEAT_PAN
    ("pan", None),
    // FEAT_LOR
    ("lor", None),
    // FEAT_VHE
    ("vh", None),
    // FEAT_PMUv3
    ("pmuv3", None),
    // FEAT_SPE
    ("spe", None),
    ("v8.1a", Some(sym::aarch64_ver_target_feature)),
    ("v8.2a", Some(sym::aarch64_ver_target_feature)),
    ("v8.3a", Some(sym::aarch64_ver_target_feature)),
    ("v8.4a", Some(sym::aarch64_ver_target_feature)),
    ("v8.5a", Some(sym::aarch64_ver_target_feature)),
    ("v8.6a", Some(sym::aarch64_ver_target_feature)),
    ("v8.7a", Some(sym::aarch64_ver_target_feature)),
];

const AARCH64_TIED_FEATURES: &[&[&str]] = &[
    &["paca", "pacg"], // Together these represent `pauth` in LLVM
];

const X86_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
    ("adx", None),
    ("aes", None),
    ("avx", None),
    ("avx2", None),
    ("avx512bf16", Some(sym::avx512_target_feature)),
    ("avx512bitalg", Some(sym::avx512_target_feature)),
    ("avx512bw", Some(sym::avx512_target_feature)),
    ("avx512cd", Some(sym::avx512_target_feature)),
    ("avx512dq", Some(sym::avx512_target_feature)),
    ("avx512er", Some(sym::avx512_target_feature)),
    ("avx512f", Some(sym::avx512_target_feature)),
    ("avx512gfni", Some(sym::avx512_target_feature)),
    ("avx512ifma", Some(sym::avx512_target_feature)),
    ("avx512pf", Some(sym::avx512_target_feature)),
    ("avx512vaes", Some(sym::avx512_target_feature)),
    ("avx512vbmi", Some(sym::avx512_target_feature)),
    ("avx512vbmi2", Some(sym::avx512_target_feature)),
    ("avx512vl", Some(sym::avx512_target_feature)),
    ("avx512vnni", Some(sym::avx512_target_feature)),
    ("avx512vp2intersect", Some(sym::avx512_target_feature)),
    ("avx512vpclmulqdq", Some(sym::avx512_target_feature)),
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

const MIPS_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
    ("fp64", Some(sym::mips_target_feature)),
    ("msa", Some(sym::mips_target_feature)),
    ("virt", Some(sym::mips_target_feature)),
];

const RISCV_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
    ("m", Some(sym::riscv_target_feature)),
    ("a", Some(sym::riscv_target_feature)),
    ("c", Some(sym::riscv_target_feature)),
    ("f", Some(sym::riscv_target_feature)),
    ("d", Some(sym::riscv_target_feature)),
    ("e", Some(sym::riscv_target_feature)),
    ("v", Some(sym::riscv_target_feature)),
    ("zfinx", Some(sym::riscv_target_feature)),
    ("zdinx", Some(sym::riscv_target_feature)),
    ("zhinx", Some(sym::riscv_target_feature)),
    ("zhinxmin", Some(sym::riscv_target_feature)),
    ("zfh", Some(sym::riscv_target_feature)),
    ("zfhmin", Some(sym::riscv_target_feature)),
    ("zba", Some(sym::riscv_target_feature)),
    ("zbb", Some(sym::riscv_target_feature)),
    ("zbc", Some(sym::riscv_target_feature)),
    ("zbs", Some(sym::riscv_target_feature)),
    ("zbkb", Some(sym::riscv_target_feature)),
    ("zbkc", Some(sym::riscv_target_feature)),
    ("zbkx", Some(sym::riscv_target_feature)),
    ("zknd", Some(sym::riscv_target_feature)),
    ("zkne", Some(sym::riscv_target_feature)),
    ("zknh", Some(sym::riscv_target_feature)),
    ("zksed", Some(sym::riscv_target_feature)),
    ("zksh", Some(sym::riscv_target_feature)),
    ("zkr", Some(sym::riscv_target_feature)),
    ("zkn", Some(sym::riscv_target_feature)),
    ("zks", Some(sym::riscv_target_feature)),
    ("zk", Some(sym::riscv_target_feature)),
    ("zkt", Some(sym::riscv_target_feature)),
];

const WASM_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[
    ("simd128", None),
    ("atomics", Some(sym::wasm_target_feature)),
    ("nontrapping-fptoint", Some(sym::wasm_target_feature)),
    ("bulk-memory", Some(sym::wasm_target_feature)),
    ("mutable-globals", Some(sym::wasm_target_feature)),
    ("reference-types", Some(sym::wasm_target_feature)),
    ("sign-ext", Some(sym::wasm_target_feature)),
];

const BPF_ALLOWED_FEATURES: &[(&str, Option<Symbol>)] = &[("alu32", Some(sym::bpf_target_feature))];

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
        .chain(BPF_ALLOWED_FEATURES.iter())
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
        "wasm32" | "wasm64" => WASM_ALLOWED_FEATURES,
        "bpf" => BPF_ALLOWED_FEATURES,
        _ => &[],
    }
}

pub fn tied_target_features(sess: &Session) -> &'static [&'static [&'static str]] {
    match &*sess.target.arch {
        "aarch64" => AARCH64_TIED_FEATURES,
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
