//! Stuff for handling `#[target_feature]` (needed for unsafe check).

use std::sync::LazyLock;

use hir_def::attr::Attrs;
use hir_def::tt;
use intern::{Symbol, sym};
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug, Default)]
pub struct TargetFeatures {
    pub(crate) enabled: FxHashSet<Symbol>,
}

impl TargetFeatures {
    pub fn from_attrs(attrs: &Attrs) -> Self {
        let mut result = TargetFeatures::from_attrs_no_implications(attrs);
        result.expand_implications();
        result
    }

    fn expand_implications(&mut self) {
        let all_implications = LazyLock::force(&TARGET_FEATURE_IMPLICATIONS);
        let mut queue = self.enabled.iter().cloned().collect::<Vec<_>>();
        while let Some(feature) = queue.pop() {
            if let Some(implications) = all_implications.get(&feature) {
                for implication in implications {
                    if self.enabled.insert(implication.clone()) {
                        queue.push(implication.clone());
                    }
                }
            }
        }
    }

    /// Retrieves the target features from the attributes, and does not expand the target features implied by them.
    pub(crate) fn from_attrs_no_implications(attrs: &Attrs) -> Self {
        let enabled = attrs
            .by_key(sym::target_feature)
            .tt_values()
            .filter_map(|tt| match tt.token_trees().flat_tokens() {
                [
                    tt::TokenTree::Leaf(tt::Leaf::Ident(enable_ident)),
                    tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: '=', .. })),
                    tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal {
                        kind: tt::LitKind::Str,
                        symbol: features,
                        ..
                    })),
                ] if enable_ident.sym == sym::enable => Some(features),
                _ => None,
            })
            .flat_map(|features| features.as_str().split(',').map(Symbol::intern))
            .collect();
        Self { enabled }
    }
}

// List of the target features each target feature implies.
// Ideally we'd depend on rustc for this, but rustc_target doesn't compile on stable,
// and t-compiler prefers for it to stay this way.

static TARGET_FEATURE_IMPLICATIONS: LazyLock<FxHashMap<Symbol, Box<[Symbol]>>> =
    LazyLock::new(|| {
        let mut result = FxHashMap::<Symbol, FxHashSet<Symbol>>::default();
        for &(feature_str, implications) in TARGET_FEATURE_IMPLICATIONS_RAW {
            let feature = Symbol::intern(feature_str);
            let implications = implications.iter().copied().map(Symbol::intern);
            // Some target features appear in two archs, e.g. Arm and x86.
            // Sometimes they contain different implications, e.g. `aes`.
            // We should probably choose by the active arch, but for now just merge them.
            result.entry(feature).or_default().extend(implications);
        }
        let mut result = result
            .into_iter()
            .map(|(feature, implications)| (feature, Box::from_iter(implications)))
            .collect::<FxHashMap<_, _>>();
        result.shrink_to_fit();
        result
    });

// spellchecker:off
const TARGET_FEATURE_IMPLICATIONS_RAW: &[(&str, &[&str])] = &[
    // Arm
    ("aes", &["neon"]),
    ("dotprod", &["neon"]),
    ("fp-armv8", &["vfp4"]),
    ("fp16", &["neon"]),
    ("i8mm", &["neon"]),
    ("neon", &["vfp3"]),
    ("sha2", &["neon"]),
    ("v6", &["v5te"]),
    ("v6k", &["v6"]),
    ("v6t2", &["v6k", "thumb2"]),
    ("v7", &["v6t2"]),
    ("v8", &["v7"]),
    ("vfp3", &["vfp2", "d32"]),
    ("vfp4", &["vfp3"]),
    // Aarch64
    ("aes", &["neon"]),
    ("dotprod", &["neon"]),
    ("dpb2", &["dpb"]),
    ("f32mm", &["sve"]),
    ("f64mm", &["sve"]),
    ("fcma", &["neon"]),
    ("fhm", &["fp16"]),
    ("fp16", &["neon"]),
    ("fp8", &["faminmax", "lut", "bf16"]),
    ("fp8dot2", &["fp8dot4"]),
    ("fp8dot4", &["fp8fma"]),
    ("fp8fma", &["fp8"]),
    ("jsconv", &["neon"]),
    ("lse128", &["lse"]),
    ("rcpc2", &["rcpc"]),
    ("rcpc3", &["rcpc2"]),
    ("rdm", &["neon"]),
    ("sha2", &["neon"]),
    ("sha3", &["sha2"]),
    ("sm4", &["neon"]),
    ("sme", &["bf16"]),
    ("sme-b16b16", &["bf16", "sme2", "sve-b16b16"]),
    ("sme-f16f16", &["sme2"]),
    ("sme-f64f64", &["sme"]),
    ("sme-f8f16", &["sme-f8f32"]),
    ("sme-f8f32", &["sme2", "fp8"]),
    ("sme-fa64", &["sme", "sve2"]),
    ("sme-i16i64", &["sme"]),
    ("sme2", &["sme"]),
    ("sme2p1", &["sme2"]),
    ("ssve-fp8dot2", &["ssve-fp8dot4"]),
    ("ssve-fp8dot4", &["ssve-fp8fma"]),
    ("ssve-fp8fma", &["sme2", "fp8"]),
    ("sve", &["neon"]),
    ("sve-b16b16", &["bf16"]),
    ("sve2", &["sve"]),
    ("sve2-aes", &["sve2", "aes"]),
    ("sve2-bitperm", &["sve2"]),
    ("sve2-sha3", &["sve2", "sha3"]),
    ("sve2-sm4", &["sve2", "sm4"]),
    ("sve2p1", &["sve2"]),
    ("v8.1a", &["crc", "lse", "rdm", "pan", "lor", "vh"]),
    ("v8.2a", &["v8.1a", "ras", "dpb"]),
    ("v8.3a", &["v8.2a", "rcpc", "paca", "pacg", "jsconv"]),
    ("v8.4a", &["v8.3a", "dotprod", "dit", "flagm"]),
    ("v8.5a", &["v8.4a", "ssbs", "sb", "dpb2", "bti"]),
    ("v8.6a", &["v8.5a", "bf16", "i8mm"]),
    ("v8.7a", &["v8.6a", "wfxt"]),
    ("v8.8a", &["v8.7a", "hbc", "mops"]),
    ("v8.9a", &["v8.8a", "cssc"]),
    ("v9.1a", &["v9a", "v8.6a"]),
    ("v9.2a", &["v9.1a", "v8.7a"]),
    ("v9.3a", &["v9.2a", "v8.8a"]),
    ("v9.4a", &["v9.3a", "v8.9a"]),
    ("v9.5a", &["v9.4a"]),
    ("v9a", &["v8.5a", "sve2"]),
    // x86
    ("aes", &["sse2"]),
    ("amx-bf16", &["amx-tile"]),
    ("amx-complex", &["amx-tile"]),
    ("amx-fp16", &["amx-tile"]),
    ("amx-int8", &["amx-tile"]),
    ("avx", &["sse4.2"]),
    ("avx2", &["avx"]),
    ("avx512bf16", &["avx512bw"]),
    ("avx512bitalg", &["avx512bw"]),
    ("avx512bw", &["avx512f"]),
    ("avx512cd", &["avx512f"]),
    ("avx512dq", &["avx512f"]),
    ("avx512f", &["avx2", "fma", "f16c"]),
    ("avx512fp16", &["avx512bw", "avx512vl", "avx512dq"]),
    ("avx512ifma", &["avx512f"]),
    ("avx512vbmi", &["avx512bw"]),
    ("avx512vbmi2", &["avx512bw"]),
    ("avx512vl", &["avx512f"]),
    ("avx512vnni", &["avx512f"]),
    ("avx512vp2intersect", &["avx512f"]),
    ("avx512vpopcntdq", &["avx512f"]),
    ("avxifma", &["avx2"]),
    ("avxneconvert", &["avx2"]),
    ("avxvnni", &["avx2"]),
    ("avxvnniint16", &["avx2"]),
    ("avxvnniint8", &["avx2"]),
    ("f16c", &["avx"]),
    ("fma", &["avx"]),
    ("gfni", &["sse2"]),
    ("kl", &["sse2"]),
    ("pclmulqdq", &["sse2"]),
    ("sha", &["sse2"]),
    ("sha512", &["avx2"]),
    ("sm3", &["avx"]),
    ("sm4", &["avx2"]),
    ("sse2", &["sse"]),
    ("sse3", &["sse2"]),
    ("sse4.1", &["ssse3"]),
    ("sse4.2", &["sse4.1"]),
    ("sse4a", &["sse3"]),
    ("ssse3", &["sse3"]),
    ("vaes", &["avx2", "aes"]),
    ("vpclmulqdq", &["avx", "pclmulqdq"]),
    ("widekl", &["kl"]),
    ("xop", &[/*"fma4", */ "avx", "sse4a"]),
    ("xsavec", &["xsave"]),
    ("xsaveopt", &["xsave"]),
    ("xsaves", &["xsave"]),
    // Hexagon
    ("hvx-length128b", &["hvx"]),
    // PowerPC
    ("power10-vector", &["power9-vector"]),
    ("power8-altivec", &["altivec"]),
    ("power8-crypto", &["power8-altivec"]),
    ("power8-vector", &["vsx", "power8-altivec"]),
    ("power9-altivec", &["power8-altivec"]),
    ("power9-vector", &["power8-vector", "power9-altivec"]),
    ("vsx", &["altivec"]),
    // MIPS
    // RISC-V
    ("a", &["zaamo", "zalrsc"]),
    ("d", &["f"]),
    ("zabha", &["zaamo"]),
    ("zdinx", &["zfinx"]),
    ("zfh", &["zfhmin"]),
    ("zfhmin", &["f"]),
    ("zhinx", &["zhinxmin"]),
    ("zhinxmin", &["zfinx"]),
    ("zk", &["zkn", "zkr", "zkt"]),
    ("zkn", &["zbkb", "zbkc", "zbkx", "zkne", "zknd", "zknh"]),
    ("zks", &["zbkb", "zbkc", "zbkx", "zksed", "zksh"]),
    // WASM
    ("relaxed-simd", &["simd128"]),
    // BPF
    ("alu32", &[]),
    // CSKY
    ("10e60", &["7e10"]),
    ("2e3", &["e2"]),
    ("3e3r2", &["3e3r1", "doloop"]),
    ("3e3r3", &["doloop"]),
    ("3e7", &["2e3"]),
    ("7e10", &["3e7"]),
    ("e1", &["elrw"]),
    ("e2", &["e2"]),
    ("mp", &["2e3"]),
    ("mp1e2", &["3e7"]),
    // LoongArch
    ("d", &["f"]),
    ("lasx", &["lsx"]),
    ("lsx", &["d"]),
    // IBM Z
    ("nnp-assist", &["vector"]),
    ("vector-enhancements-1", &["vector"]),
    ("vector-enhancements-2", &["vector-enhancements-1"]),
    ("vector-packed-decimal", &["vector"]),
    ("vector-packed-decimal-enhancement", &["vector-packed-decimal"]),
    ("vector-packed-decimal-enhancement-2", &["vector-packed-decimal-enhancement"]),
    // SPARC
    // m68k
    ("isa-68010", &["isa-68000"]),
    ("isa-68020", &["isa-68010"]),
    ("isa-68030", &["isa-68020"]),
    ("isa-68040", &["isa-68030", "isa-68882"]),
    ("isa-68060", &["isa-68040"]),
    ("isa-68882", &["isa-68881"]),
];
// spellchecker:on
