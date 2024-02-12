use rustc_span::symbol::sym;
use rustc_span::symbol::Symbol;

/// Features that control behaviour of rustc, rather than the codegen.
pub const RUSTC_SPECIFIC_FEATURES: &[&str] = &["crt-static"];

/// Stability information for target features.
#[derive(Debug, Clone, Copy)]
pub enum Stability {
    /// This target feature is stable, it can be used in `#[target_feature]` and
    /// `#[cfg(target_feature)]`.
    Stable,
    /// This target feature is unstable; using it in `#[target_feature]` or `#[cfg(target_feature)]`
    /// requires enabling the given nightly feature.
    Unstable(Symbol),
}
use Stability::*;

impl Stability {
    pub fn as_feature_name(self) -> Option<Symbol> {
        match self {
            Stable => None,
            Unstable(s) => Some(s),
        }
    }

    pub fn is_stable(self) -> bool {
        matches!(self, Stable)
    }
}

// Here we list target features that rustc "understands": they can be used in `#[target_feature]`
// and `#[cfg(target_feature)]`. They also do not trigger any warnings when used with
// `-Ctarget-feature`.
//
// When adding features to the below lists
// check whether they're named already elsewhere in rust
// e.g. in stdarch and whether the given name matches LLVM's
// if it doesn't, to_llvm_feature in llvm_util in rustc_codegen_llvm needs to be adapted.
//
// Also note that all target features listed here must be purely additive: for target_feature 1.1 to
// be sound, we can never allow features like `+soft-float` (on x86) to be controlled on a
// per-function level, since we would then allow safe calls from functions with `+soft-float` to
// functions without that feature!
//
// When adding a new feature, be particularly mindful of features that affect function ABIs. Those
// need to be treated very carefully to avoid introducing unsoundness! This often affects features
// that enable/disable hardfloat support (see https://github.com/rust-lang/rust/issues/116344 for an
// example of this going wrong), but features enabling new SIMD registers are also a concern (see
// https://github.com/rust-lang/rust/issues/116558 for an example of this going wrong).
//
// Stabilizing a target feature requires t-lang approval.

const ARM_ALLOWED_FEATURES: &[(&str, Stability)] = &[
    // tidy-alphabetical-start
    ("aclass", Unstable(sym::arm_target_feature)),
    ("aes", Unstable(sym::arm_target_feature)),
    ("crc", Unstable(sym::arm_target_feature)),
    ("d32", Unstable(sym::arm_target_feature)),
    ("dotprod", Unstable(sym::arm_target_feature)),
    ("dsp", Unstable(sym::arm_target_feature)),
    ("fp-armv8", Unstable(sym::arm_target_feature)),
    ("i8mm", Unstable(sym::arm_target_feature)),
    ("mclass", Unstable(sym::arm_target_feature)),
    ("neon", Unstable(sym::arm_target_feature)),
    ("rclass", Unstable(sym::arm_target_feature)),
    ("sha2", Unstable(sym::arm_target_feature)),
    // This is needed for inline assembly, but shouldn't be stabilized as-is
    // since it should be enabled per-function using #[instruction_set], not
    // #[target_feature].
    ("thumb-mode", Unstable(sym::arm_target_feature)),
    ("thumb2", Unstable(sym::arm_target_feature)),
    ("trustzone", Unstable(sym::arm_target_feature)),
    ("v5te", Unstable(sym::arm_target_feature)),
    ("v6", Unstable(sym::arm_target_feature)),
    ("v6k", Unstable(sym::arm_target_feature)),
    ("v6t2", Unstable(sym::arm_target_feature)),
    ("v7", Unstable(sym::arm_target_feature)),
    ("v8", Unstable(sym::arm_target_feature)),
    ("vfp2", Unstable(sym::arm_target_feature)),
    ("vfp3", Unstable(sym::arm_target_feature)),
    ("vfp4", Unstable(sym::arm_target_feature)),
    ("virtualization", Unstable(sym::arm_target_feature)),
    // tidy-alphabetical-end
];

const AARCH64_ALLOWED_FEATURES: &[(&str, Stability)] = &[
    // tidy-alphabetical-start
    // FEAT_AES
    ("aes", Stable),
    // FEAT_BF16
    ("bf16", Stable),
    // FEAT_BTI
    ("bti", Stable),
    // FEAT_CRC
    ("crc", Stable),
    // FEAT_DIT
    ("dit", Stable),
    // FEAT_DotProd
    ("dotprod", Stable),
    // FEAT_DPB
    ("dpb", Stable),
    // FEAT_DPB2
    ("dpb2", Stable),
    // FEAT_F32MM
    ("f32mm", Stable),
    // FEAT_F64MM
    ("f64mm", Stable),
    // FEAT_FCMA
    ("fcma", Stable),
    // FEAT_FHM
    ("fhm", Stable),
    // FEAT_FLAGM
    ("flagm", Stable),
    // FEAT_FP16
    ("fp16", Stable),
    // FEAT_FRINTTS
    ("frintts", Stable),
    // FEAT_I8MM
    ("i8mm", Stable),
    // FEAT_JSCVT
    ("jsconv", Stable),
    // FEAT_LOR
    ("lor", Stable),
    // FEAT_LSE
    ("lse", Stable),
    // FEAT_MTE
    ("mte", Stable),
    // FEAT_AdvSimd & FEAT_FP
    ("neon", Stable),
    // FEAT_PAUTH (address authentication)
    ("paca", Stable),
    // FEAT_PAUTH (generic authentication)
    ("pacg", Stable),
    // FEAT_PAN
    ("pan", Stable),
    // FEAT_PMUv3
    ("pmuv3", Stable),
    // FEAT_RAND
    ("rand", Stable),
    // FEAT_RAS
    ("ras", Stable),
    // FEAT_RCPC
    ("rcpc", Stable),
    // FEAT_RCPC2
    ("rcpc2", Stable),
    // FEAT_RDM
    ("rdm", Stable),
    // FEAT_SB
    ("sb", Stable),
    // FEAT_SHA1 & FEAT_SHA256
    ("sha2", Stable),
    // FEAT_SHA512 & FEAT_SHA3
    ("sha3", Stable),
    // FEAT_SM3 & FEAT_SM4
    ("sm4", Stable),
    // FEAT_SPE
    ("spe", Stable),
    // FEAT_SSBS
    ("ssbs", Stable),
    // FEAT_SVE
    ("sve", Stable),
    // FEAT_SVE2
    ("sve2", Stable),
    // FEAT_SVE2_AES
    ("sve2-aes", Stable),
    // FEAT_SVE2_BitPerm
    ("sve2-bitperm", Stable),
    // FEAT_SVE2_SHA3
    ("sve2-sha3", Stable),
    // FEAT_SVE2_SM4
    ("sve2-sm4", Stable),
    // FEAT_TME
    ("tme", Stable),
    ("v8.1a", Unstable(sym::aarch64_ver_target_feature)),
    ("v8.2a", Unstable(sym::aarch64_ver_target_feature)),
    ("v8.3a", Unstable(sym::aarch64_ver_target_feature)),
    ("v8.4a", Unstable(sym::aarch64_ver_target_feature)),
    ("v8.5a", Unstable(sym::aarch64_ver_target_feature)),
    ("v8.6a", Unstable(sym::aarch64_ver_target_feature)),
    ("v8.7a", Unstable(sym::aarch64_ver_target_feature)),
    // FEAT_VHE
    ("vh", Stable),
    // tidy-alphabetical-end
];

const AARCH64_TIED_FEATURES: &[&[&str]] = &[
    &["paca", "pacg"], // Together these represent `pauth` in LLVM
];

const X86_ALLOWED_FEATURES: &[(&str, Stability)] = &[
    // tidy-alphabetical-start
    ("adx", Stable),
    ("aes", Stable),
    ("avx", Stable),
    ("avx2", Stable),
    ("avx512bf16", Unstable(sym::avx512_target_feature)),
    ("avx512bitalg", Unstable(sym::avx512_target_feature)),
    ("avx512bw", Unstable(sym::avx512_target_feature)),
    ("avx512cd", Unstable(sym::avx512_target_feature)),
    ("avx512dq", Unstable(sym::avx512_target_feature)),
    ("avx512er", Unstable(sym::avx512_target_feature)),
    ("avx512f", Unstable(sym::avx512_target_feature)),
    ("avx512fp16", Unstable(sym::avx512_target_feature)),
    ("avx512ifma", Unstable(sym::avx512_target_feature)),
    ("avx512pf", Unstable(sym::avx512_target_feature)),
    ("avx512vbmi", Unstable(sym::avx512_target_feature)),
    ("avx512vbmi2", Unstable(sym::avx512_target_feature)),
    ("avx512vl", Unstable(sym::avx512_target_feature)),
    ("avx512vnni", Unstable(sym::avx512_target_feature)),
    ("avx512vp2intersect", Unstable(sym::avx512_target_feature)),
    ("avx512vpopcntdq", Unstable(sym::avx512_target_feature)),
    ("bmi1", Stable),
    ("bmi2", Stable),
    ("cmpxchg16b", Stable),
    ("ermsb", Unstable(sym::ermsb_target_feature)),
    ("f16c", Stable),
    ("fma", Stable),
    ("fxsr", Stable),
    ("gfni", Unstable(sym::avx512_target_feature)),
    ("lahfsahf", Unstable(sym::lahfsahf_target_feature)),
    ("lzcnt", Stable),
    ("movbe", Stable),
    ("pclmulqdq", Stable),
    ("popcnt", Stable),
    ("prfchw", Unstable(sym::prfchw_target_feature)),
    ("rdrand", Stable),
    ("rdseed", Stable),
    ("rtm", Unstable(sym::rtm_target_feature)),
    ("sha", Stable),
    ("sse", Stable),
    ("sse2", Stable),
    ("sse3", Stable),
    ("sse4.1", Stable),
    ("sse4.2", Stable),
    ("sse4a", Unstable(sym::sse4a_target_feature)),
    ("ssse3", Stable),
    ("tbm", Unstable(sym::tbm_target_feature)),
    ("vaes", Unstable(sym::avx512_target_feature)),
    ("vpclmulqdq", Unstable(sym::avx512_target_feature)),
    ("xsave", Stable),
    ("xsavec", Stable),
    ("xsaveopt", Stable),
    ("xsaves", Stable),
    // tidy-alphabetical-end
];

const HEXAGON_ALLOWED_FEATURES: &[(&str, Stability)] = &[
    // tidy-alphabetical-start
    ("hvx", Unstable(sym::hexagon_target_feature)),
    ("hvx-length128b", Unstable(sym::hexagon_target_feature)),
    // tidy-alphabetical-end
];

const POWERPC_ALLOWED_FEATURES: &[(&str, Stability)] = &[
    // tidy-alphabetical-start
    ("altivec", Unstable(sym::powerpc_target_feature)),
    ("power10-vector", Unstable(sym::powerpc_target_feature)),
    ("power8-altivec", Unstable(sym::powerpc_target_feature)),
    ("power8-vector", Unstable(sym::powerpc_target_feature)),
    ("power9-altivec", Unstable(sym::powerpc_target_feature)),
    ("power9-vector", Unstable(sym::powerpc_target_feature)),
    ("vsx", Unstable(sym::powerpc_target_feature)),
    // tidy-alphabetical-end
];

const MIPS_ALLOWED_FEATURES: &[(&str, Stability)] = &[
    // tidy-alphabetical-start
    ("fp64", Unstable(sym::mips_target_feature)),
    ("msa", Unstable(sym::mips_target_feature)),
    ("virt", Unstable(sym::mips_target_feature)),
    // tidy-alphabetical-end
];

const RISCV_ALLOWED_FEATURES: &[(&str, Stability)] = &[
    // tidy-alphabetical-start
    ("a", Stable),
    ("c", Stable),
    ("d", Unstable(sym::riscv_target_feature)),
    ("e", Unstable(sym::riscv_target_feature)),
    ("f", Unstable(sym::riscv_target_feature)),
    ("fast-unaligned-access", Unstable(sym::riscv_target_feature)),
    ("m", Stable),
    ("relax", Unstable(sym::riscv_target_feature)),
    ("v", Unstable(sym::riscv_target_feature)),
    ("zba", Stable),
    ("zbb", Stable),
    ("zbc", Stable),
    ("zbkb", Stable),
    ("zbkc", Stable),
    ("zbkx", Stable),
    ("zbs", Stable),
    ("zdinx", Unstable(sym::riscv_target_feature)),
    ("zfh", Unstable(sym::riscv_target_feature)),
    ("zfhmin", Unstable(sym::riscv_target_feature)),
    ("zfinx", Unstable(sym::riscv_target_feature)),
    ("zhinx", Unstable(sym::riscv_target_feature)),
    ("zhinxmin", Unstable(sym::riscv_target_feature)),
    ("zk", Stable),
    ("zkn", Stable),
    ("zknd", Stable),
    ("zkne", Stable),
    ("zknh", Stable),
    ("zkr", Stable),
    ("zks", Stable),
    ("zksed", Stable),
    ("zksh", Stable),
    ("zkt", Stable),
    // tidy-alphabetical-end
];

const WASM_ALLOWED_FEATURES: &[(&str, Stability)] = &[
    // tidy-alphabetical-start
    ("atomics", Unstable(sym::wasm_target_feature)),
    ("bulk-memory", Unstable(sym::wasm_target_feature)),
    ("exception-handling", Unstable(sym::wasm_target_feature)),
    ("multivalue", Unstable(sym::wasm_target_feature)),
    ("mutable-globals", Unstable(sym::wasm_target_feature)),
    ("nontrapping-fptoint", Unstable(sym::wasm_target_feature)),
    ("reference-types", Unstable(sym::wasm_target_feature)),
    ("relaxed-simd", Unstable(sym::wasm_target_feature)),
    ("sign-ext", Unstable(sym::wasm_target_feature)),
    ("simd128", Stable),
    // tidy-alphabetical-end
];

const BPF_ALLOWED_FEATURES: &[(&str, Stability)] = &[("alu32", Unstable(sym::bpf_target_feature))];

const CSKY_ALLOWED_FEATURES: &[(&str, Stability)] = &[
    // tidy-alphabetical-start
    ("10e60", Unstable(sym::csky_target_feature)),
    ("2e3", Unstable(sym::csky_target_feature)),
    ("3e3r1", Unstable(sym::csky_target_feature)),
    ("3e3r2", Unstable(sym::csky_target_feature)),
    ("3e3r3", Unstable(sym::csky_target_feature)),
    ("3e7", Unstable(sym::csky_target_feature)),
    ("7e10", Unstable(sym::csky_target_feature)),
    ("cache", Unstable(sym::csky_target_feature)),
    ("doloop", Unstable(sym::csky_target_feature)),
    ("dsp1e2", Unstable(sym::csky_target_feature)),
    ("dspe60", Unstable(sym::csky_target_feature)),
    ("e1", Unstable(sym::csky_target_feature)),
    ("e2", Unstable(sym::csky_target_feature)),
    ("edsp", Unstable(sym::csky_target_feature)),
    ("elrw", Unstable(sym::csky_target_feature)),
    ("float1e2", Unstable(sym::csky_target_feature)),
    ("float1e3", Unstable(sym::csky_target_feature)),
    ("float3e4", Unstable(sym::csky_target_feature)),
    ("float7e60", Unstable(sym::csky_target_feature)),
    ("floate1", Unstable(sym::csky_target_feature)),
    ("hard-tp", Unstable(sym::csky_target_feature)),
    ("high-registers", Unstable(sym::csky_target_feature)),
    ("hwdiv", Unstable(sym::csky_target_feature)),
    ("mp", Unstable(sym::csky_target_feature)),
    ("mp1e2", Unstable(sym::csky_target_feature)),
    ("nvic", Unstable(sym::csky_target_feature)),
    ("trust", Unstable(sym::csky_target_feature)),
    ("vdsp2e60f", Unstable(sym::csky_target_feature)),
    ("vdspv1", Unstable(sym::csky_target_feature)),
    ("vdspv2", Unstable(sym::csky_target_feature)),
    // tidy-alphabetical-end
    //fpu
    // tidy-alphabetical-start
    ("fdivdu", Unstable(sym::csky_target_feature)),
    ("fpuv2_df", Unstable(sym::csky_target_feature)),
    ("fpuv2_sf", Unstable(sym::csky_target_feature)),
    ("fpuv3_df", Unstable(sym::csky_target_feature)),
    ("fpuv3_hf", Unstable(sym::csky_target_feature)),
    ("fpuv3_hi", Unstable(sym::csky_target_feature)),
    ("fpuv3_sf", Unstable(sym::csky_target_feature)),
    ("hard-float", Unstable(sym::csky_target_feature)),
    ("hard-float-abi", Unstable(sym::csky_target_feature)),
    // tidy-alphabetical-end
];

const LOONGARCH_ALLOWED_FEATURES: &[(&str, Stability)] = &[
    // tidy-alphabetical-start
    ("d", Unstable(sym::loongarch_target_feature)),
    ("f", Unstable(sym::loongarch_target_feature)),
    ("lasx", Unstable(sym::loongarch_target_feature)),
    ("lbt", Unstable(sym::loongarch_target_feature)),
    ("lsx", Unstable(sym::loongarch_target_feature)),
    ("lvz", Unstable(sym::loongarch_target_feature)),
    ("ual", Unstable(sym::loongarch_target_feature)),
    // tidy-alphabetical-end
];

/// When rustdoc is running, provide a list of all known features so that all their respective
/// primitives may be documented.
///
/// IMPORTANT: If you're adding another feature list above, make sure to add it to this iterator!
pub fn all_known_features() -> impl Iterator<Item = (&'static str, Stability)> {
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
        .chain(CSKY_ALLOWED_FEATURES)
        .chain(LOONGARCH_ALLOWED_FEATURES)
        .cloned()
}

impl super::spec::Target {
    pub fn supported_target_features(&self) -> &'static [(&'static str, Stability)] {
        match &*self.arch {
            "arm" => ARM_ALLOWED_FEATURES,
            "aarch64" => AARCH64_ALLOWED_FEATURES,
            "x86" | "x86_64" => X86_ALLOWED_FEATURES,
            "hexagon" => HEXAGON_ALLOWED_FEATURES,
            "mips" | "mips32r6" | "mips64" | "mips64r6" => MIPS_ALLOWED_FEATURES,
            "powerpc" | "powerpc64" => POWERPC_ALLOWED_FEATURES,
            "riscv32" | "riscv64" => RISCV_ALLOWED_FEATURES,
            "wasm32" | "wasm64" => WASM_ALLOWED_FEATURES,
            "bpf" => BPF_ALLOWED_FEATURES,
            "csky" => CSKY_ALLOWED_FEATURES,
            "loongarch64" => LOONGARCH_ALLOWED_FEATURES,
            _ => &[],
        }
    }

    pub fn tied_target_features(&self) -> &'static [&'static [&'static str]] {
        match &*self.arch {
            "aarch64" => AARCH64_TIED_FEATURES,
            _ => &[],
        }
    }
}
