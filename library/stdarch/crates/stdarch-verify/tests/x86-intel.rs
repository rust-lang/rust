#![allow(bad_style)]
#![allow(unused)]
#![allow(
    clippy::shadow_reuse,
    clippy::cast_lossless,
    clippy::match_same_arms,
    clippy::nonminimal_bool,
    clippy::print_stdout,
    clippy::use_debug,
    clippy::eq_op,
    clippy::useless_format
)]

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;

const PRINT_INSTRUCTION_VIOLATIONS: bool = false;
const PRINT_MISSING_LISTS: bool = false;
const PRINT_MISSING_LISTS_MARKDOWN: bool = false;

struct Function {
    name: &'static str,
    arguments: &'static [&'static Type],
    ret: Option<&'static Type>,
    target_feature: Option<&'static str>,
    instrs: &'static [&'static str],
    file: &'static str,
    required_const: &'static [usize],
    has_test: bool,
}

static F32: Type = Type::PrimFloat(32);
static F64: Type = Type::PrimFloat(64);
static I8: Type = Type::PrimSigned(8);
static I16: Type = Type::PrimSigned(16);
static I32: Type = Type::PrimSigned(32);
static I64: Type = Type::PrimSigned(64);
static U8: Type = Type::PrimUnsigned(8);
static U16: Type = Type::PrimUnsigned(16);
static U32: Type = Type::PrimUnsigned(32);
static U64: Type = Type::PrimUnsigned(64);
static U128: Type = Type::PrimUnsigned(128);
static ORDERING: Type = Type::Ordering;

static M64: Type = Type::M64;
static M128: Type = Type::M128;
static M128BH: Type = Type::M128BH;
static M128I: Type = Type::M128I;
static M128D: Type = Type::M128D;
static M256: Type = Type::M256;
static M256BH: Type = Type::M256BH;
static M256I: Type = Type::M256I;
static M256D: Type = Type::M256D;
static M512: Type = Type::M512;
static M512BH: Type = Type::M512BH;
static M512I: Type = Type::M512I;
static M512D: Type = Type::M512D;
static MMASK8: Type = Type::MMASK8;
static MMASK16: Type = Type::MMASK16;
static MMASK32: Type = Type::MMASK32;
static MMASK64: Type = Type::MMASK64;
static MM_CMPINT_ENUM: Type = Type::MM_CMPINT_ENUM;
static MM_MANTISSA_NORM_ENUM: Type = Type::MM_MANTISSA_NORM_ENUM;
static MM_MANTISSA_SIGN_ENUM: Type = Type::MM_MANTISSA_SIGN_ENUM;
static MM_PERM_ENUM: Type = Type::MM_PERM_ENUM;

static TUPLE: Type = Type::Tuple;
static CPUID: Type = Type::CpuidResult;
static NEVER: Type = Type::Never;

#[derive(Debug)]
enum Type {
    PrimFloat(u8),
    PrimSigned(u8),
    PrimUnsigned(u8),
    MutPtr(&'static Type),
    ConstPtr(&'static Type),
    M64,
    M128,
    M128BH,
    M128D,
    M128I,
    M256,
    M256BH,
    M256D,
    M256I,
    M512,
    M512BH,
    M512D,
    M512I,
    MMASK8,
    MMASK16,
    MMASK32,
    MMASK64,
    MM_CMPINT_ENUM,
    MM_MANTISSA_NORM_ENUM,
    MM_MANTISSA_SIGN_ENUM,
    MM_PERM_ENUM,
    Tuple,
    CpuidResult,
    Never,
    Ordering,
}

stdarch_verify::x86_functions!(static FUNCTIONS);

#[derive(Deserialize)]
struct Data {
    #[serde(rename = "intrinsic", default)]
    intrinsics: Vec<Intrinsic>,
}

#[derive(Deserialize)]
struct Intrinsic {
    #[serde(rename = "return")]
    return_: Return,
    name: String,
    #[serde(rename = "CPUID", default)]
    cpuid: Vec<String>,
    #[serde(rename = "parameter", default)]
    parameters: Vec<Parameter>,
    #[serde(default)]
    instruction: Vec<Instruction>,
}

#[derive(Deserialize)]
struct Parameter {
    #[serde(rename = "type")]
    type_: String,
}

#[derive(Deserialize)]
struct Return {
    #[serde(rename = "type")]
    type_: String,
}

#[derive(Deserialize, Debug)]
struct Instruction {
    name: String,
}

macro_rules! bail {
    ($($t:tt)*) => (return Err(format!($($t)*)))
}

#[test]
fn verify_all_signatures() {
    // This XML document was downloaded from Intel's site. To update this you
    // can visit intel's intrinsics guide online documentation:
    //
    //   https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
    //
    // Open up the network console and you'll see an xml file was downloaded
    // (currently called data-3.4.xml). That's the file we downloaded
    // here.
    let xml = include_bytes!("../x86-intel.xml");

    let xml = &xml[..];
    let data: Data = serde_xml_rs::from_reader(xml).expect("failed to deserialize xml");
    let mut map = HashMap::new();
    for intrinsic in &data.intrinsics {
        map.entry(&intrinsic.name[..])
            .or_insert_with(Vec::new)
            .push(intrinsic);
    }

    let mut all_valid = true;
    'outer: for rust in FUNCTIONS {
        if !rust.has_test {
            // FIXME: this list should be almost empty
            let skip = [
                "__readeflags",
                "__readeflags",
                "__writeeflags",
                "__writeeflags",
                "_mm_comige_ss",
                "_mm_cvt_ss2si",
                "_mm_cvtt_ss2si",
                "_mm_cvt_si2ss",
                "_mm_set_ps1",
                "_mm_load_ps1",
                "_mm_store_ps1",
                "_mm_getcsr",
                "_mm_setcsr",
                "_MM_GET_EXCEPTION_MASK",
                "_MM_GET_EXCEPTION_STATE",
                "_MM_GET_FLUSH_ZERO_MODE",
                "_MM_GET_ROUNDING_MODE",
                "_MM_SET_EXCEPTION_MASK",
                "_MM_SET_EXCEPTION_STATE",
                "_MM_SET_FLUSH_ZERO_MODE",
                "_MM_SET_ROUNDING_MODE",
                "_mm_prefetch",
                "_mm_undefined_ps",
                "_m_pmaxsw",
                "_m_pmaxub",
                "_m_pminsw",
                "_m_pminub",
                "_m_pavgb",
                "_m_pavgw",
                "_m_psadbw",
                "_mm_cvt_pi2ps",
                "_m_maskmovq",
                "_m_pextrw",
                "_m_pinsrw",
                "_m_pmovmskb",
                "_m_pshufw",
                "_mm_cvtt_ps2pi",
                "_mm_cvt_ps2pi",
                "__cpuid_count",
                "__cpuid",
                "__get_cpuid_max",
                "_xsave",
                "_xrstor",
                "_xsetbv",
                "_xgetbv",
                "_xsaveopt",
                "_xsavec",
                "_xsaves",
                "_xrstors",
                "_mm_bslli_si128",
                "_mm_bsrli_si128",
                "_mm_undefined_pd",
                "_mm_undefined_si128",
                "_mm_cvtps_ph",
                "_mm256_cvtps_ph",
                "_rdtsc",
                "__rdtscp",
                "_mm256_castps128_ps256",
                "_mm256_castpd128_pd256",
                "_mm256_castsi128_si256",
                "_mm256_undefined_ps",
                "_mm256_undefined_pd",
                "_mm256_undefined_si256",
                "_bextr2_u32",
                "_mm_tzcnt_32",
                "_m_paddb",
                "_m_paddw",
                "_m_paddd",
                "_m_paddsb",
                "_m_paddsw",
                "_m_paddusb",
                "_m_paddusw",
                "_m_psubb",
                "_m_psubw",
                "_m_psubd",
                "_m_psubsb",
                "_m_psubsw",
                "_m_psubusb",
                "_m_psubusw",
                "_mm_set_pi16",
                "_mm_set_pi32",
                "_mm_set_pi8",
                "_mm_set1_pi16",
                "_mm_set1_pi32",
                "_mm_set1_pi8",
                "_mm_setr_pi16",
                "_mm_setr_pi32",
                "_mm_setr_pi8",
                "ud2",
                "_mm_min_epi8",
                "_mm_min_epi32",
                "_xbegin",
                "_xend",
                "_rdrand16_step",
                "_rdrand32_step",
                "_rdseed16_step",
                "_rdseed32_step",
                "_fxsave",
                "_fxrstor",
                "_t1mskc_u64",
                "_mm256_shuffle_epi32",
                "_mm256_bslli_epi128",
                "_mm256_bsrli_epi128",
                "_mm256_unpackhi_epi8",
                "_mm256_unpacklo_epi8",
                "_mm256_unpackhi_epi16",
                "_mm256_unpacklo_epi16",
                "_mm256_unpackhi_epi32",
                "_mm256_unpacklo_epi32",
                "_mm256_unpackhi_epi64",
                "_mm256_unpacklo_epi64",
                "_xsave64",
                "_xrstor64",
                "_xsaveopt64",
                "_xsavec64",
                "_xsaves64",
                "_xrstors64",
                "_mm_cvtsi64x_si128",
                "_mm_cvtsi128_si64x",
                "_mm_cvtsi64x_sd",
                "cmpxchg16b",
                "_rdrand64_step",
                "_rdseed64_step",
                "_bextr2_u64",
                "_mm_tzcnt_64",
                "_fxsave64",
                "_fxrstor64",
                "_mm512_undefined_ps",
                "_mm512_undefined_pd",
                "_mm512_undefined_epi32",
                "_mm512_undefined",
            ];
            if !skip.contains(&rust.name) {
                println!(
                    "missing run-time test named `test_{}` for `{}`",
                    {
                        let mut id = rust.name;
                        while id.starts_with('_') {
                            id = &id[1..];
                        }
                        id
                    },
                    rust.name
                );
                all_valid = false;
            }
        }

        match rust.name {
            // These aren't defined by Intel but they're defined by what appears
            // to be all other compilers. For more information see
            // rust-lang/stdarch#307, and otherwise these signatures
            // have all been manually verified.
            "__readeflags" |
            "__writeeflags" |
            "__cpuid_count" |
            "__cpuid" |
            "__get_cpuid_max" |
            // Not listed with intel, but manually verified
            "cmpxchg16b" |
            // The UD2 intrinsic is not defined by Intel, but it was agreed on
            // in the RFC Issue 2512:
            // https://github.com/rust-lang/rfcs/issues/2512
            "ud2"
                => continue,
            // Intel requires the mask argument for _mm_shuffle_ps to be an
            // unsigned integer, but all other _mm_shuffle_.. intrinsics
            // take a signed-integer. This breaks `_MM_SHUFFLE` for
            // `_mm_shuffle_ps`:
            "_mm_shuffle_ps" => continue,
            _ => {}
        }

        // these are all AMD-specific intrinsics
        if let Some(feature) = rust.target_feature {
            if feature.contains("sse4a") || feature.contains("tbm") {
                continue;
            }
        }

        let intel = match map.remove(rust.name) {
            Some(i) => i,
            None => panic!("missing intel definition for {}", rust.name),
        };

        let mut errors = Vec::new();
        for intel in intel {
            match matches(rust, intel) {
                Ok(()) => continue 'outer,
                Err(e) => errors.push(e),
            }
        }
        println!("failed to verify `{}`", rust.name);
        for error in errors {
            println!("  * {}", error);
        }
        all_valid = false;
    }
    assert!(all_valid);

    let mut missing = BTreeMap::new();
    for (name, intel) in &map {
        // currently focused mainly on missing SIMD intrinsics, but there's
        // definitely some other assorted ones that we're missing.
        if !name.starts_with("_mm") {
            continue;
        }

        // we'll get to avx-512 later
        // let avx512 = intel.iter().any(|i| {
        //     i.name.starts_with("_mm512") || i.cpuid.iter().any(|c| {
        //         c.contains("512")
        //     })
        // });
        // if avx512 {
        //     continue
        // }

        for intel in intel {
            missing
                .entry(&intel.cpuid)
                .or_insert_with(Vec::new)
                .push(intel);
        }
    }

    // generate a bulleted list of missing intrinsics
    if PRINT_MISSING_LISTS || PRINT_MISSING_LISTS_MARKDOWN {
        for (k, v) in missing {
            if PRINT_MISSING_LISTS_MARKDOWN {
                println!("\n<details><summary>{:?}</summary><p>\n", k);
                for intel in v {
                    let url = format!(
                        "https://software.intel.com/sites/landingpage\
                         /IntrinsicsGuide/#text={}&expand=5236",
                        intel.name
                    );
                    println!("  * [ ] [`{}`]({})", intel.name, url);
                }
                println!("</p></details>\n");
            } else {
                println!("\n{:?}\n", k);
                for intel in v {
                    println!("\t{}", intel.name);
                }
            }
        }
    }
}

fn matches(rust: &Function, intel: &Intrinsic) -> Result<(), String> {
    // Verify that all `#[target_feature]` annotations are correct,
    // ensuring that we've actually enabled the right instruction
    // set for this intrinsic.
    match rust.name {
        "_bswap" | "_bswap64" => {}

        // These don't actually have a target feature unlike their brethren with
        // the `x` inside the name which requires adx
        "_addcarry_u32" | "_addcarry_u64" | "_subborrow_u32" | "_subborrow_u64" => {}

        "_bittest"
        | "_bittestandset"
        | "_bittestandreset"
        | "_bittestandcomplement"
        | "_bittest64"
        | "_bittestandset64"
        | "_bittestandreset64"
        | "_bittestandcomplement64" => {}

        _ => {
            if intel.cpuid.is_empty() {
                bail!("missing cpuid for {}", rust.name);
            }
        }
    }

    for cpuid in &intel.cpuid {
        // The pause intrinsic is in the SSE2 module, but it is backwards
        // compatible with CPUs without SSE2, and it therefore does not need the
        // target-feature attribute.
        if rust.name == "_mm_pause" {
            continue;
        }
        // this is needed by _xsave and probably some related intrinsics,
        // but let's just skip it for now.
        if *cpuid == "XSS" {
            continue;
        }

        // these flags on the rdtsc/rtdscp intrinsics we don't test for right
        // now, but we may wish to add these one day!
        //
        // For more info see #308
        if *cpuid == "TSC" || *cpuid == "RDTSCP" {
            continue;
        }

        let cpuid = cpuid
            .chars()
            .flat_map(|c| c.to_lowercase())
            .collect::<String>();

        // Fix mismatching feature names:
        let fixup_cpuid = |cpuid: String| match cpuid.as_ref() {
            // The XML file names IFMA as "avx512ifma52", while Rust calls
            // it "avx512ifma".
            "avx512ifma52" => String::from("avx512ifma"),
            // The XML file names BITALG as "avx512_bitalg", while Rust calls
            // it "avx512bitalg".
            "avx512_bitalg" => String::from("avx512bitalg"),
            // The XML file names VBMI as "avx512_vbmi", while Rust calls
            // it "avx512vbmi".
            "avx512_vbmi" => String::from("avx512vbmi"),
            // The XML file names VBMI2 as "avx512_vbmi2", while Rust calls
            // it "avx512vbmi2".
            "avx512_vbmi2" => String::from("avx512vbmi2"),
            // The XML file names VNNI as "avx512_vnni", while Rust calls
            // it "avx512vnni".
            "avx512_vnni" => String::from("avx512vnni"),
            // Some AVX512f intrinsics are also supported by Knight's Corner.
            // The XML lists them as avx512f/kncni, but we are solely gating
            // them behind avx512f since we don't have a KNC feature yet.
            "avx512f/kncni" => String::from("avx512f"),
            // See: https://github.com/rust-lang/stdarch/issues/738
            // The intrinsics guide calls `f16c` `fp16c` in disagreement with
            // Intel's architecture manuals.
            "fp16c" => String::from("f16c"),
            "avx512_bf16" => String::from("avx512bf16"),
            // The XML file names VNNI as "avx512_bf16", while Rust calls
            // it "avx512bf16".
            _ => cpuid,
        };
        let fixed_cpuid = fixup_cpuid(cpuid);

        let rust_feature = rust
            .target_feature
            .unwrap_or_else(|| panic!("no target feature listed for {}", rust.name));

        if rust_feature.contains(&fixed_cpuid) {
            continue;
        }
        bail!(
            "intel cpuid `{}` not in `{}` for {}",
            fixed_cpuid,
            rust_feature,
            rust.name
        )
    }

    if PRINT_INSTRUCTION_VIOLATIONS {
        if rust.instrs.is_empty() {
            if !intel.instruction.is_empty() {
                println!(
                    "instruction not listed for `{}`, but intel lists {:?}",
                    rust.name, intel.instruction
                );
            }

        // If intel doesn't list any instructions and we do then don't
        // bother trying to look for instructions in intel, we've just got
        // some extra assertions on our end.
        } else if !intel.instruction.is_empty() {
            for instr in rust.instrs {
                let asserting = intel.instruction.iter().any(|a| a.name.starts_with(instr));
                if !asserting {
                    println!(
                        "intel failed to list `{}` as an instruction for `{}`",
                        instr, rust.name
                    );
                }
            }
        }
    }

    // Make sure we've got the right return type.
    if let Some(t) = rust.ret {
        equate(t, &intel.return_.type_, rust.name, false)?;
    } else if intel.return_.type_ != "" && intel.return_.type_ != "void" {
        bail!(
            "{} returns `{}` with intel, void in rust",
            rust.name,
            intel.return_.type_
        )
    }

    // If there's no arguments on Rust's side intel may list one "void"
    // argument, so handle that here.
    if rust.arguments.is_empty() && intel.parameters.len() == 1 {
        if intel.parameters[0].type_ != "void" {
            bail!("rust has 0 arguments, intel has one for")
        }
    } else {
        // Otherwise we want all parameters to be exactly the same
        if rust.arguments.len() != intel.parameters.len() {
            bail!("wrong number of arguments on {}", rust.name)
        }
        for (i, (a, b)) in intel.parameters.iter().zip(rust.arguments).enumerate() {
            let is_const = rust.required_const.contains(&i);
            equate(b, &a.type_, &intel.name, is_const)?;
        }
    }

    let any_i64 = rust
        .arguments
        .iter()
        .cloned()
        .chain(rust.ret)
        .any(|arg| matches!(*arg, Type::PrimSigned(64) | Type::PrimUnsigned(64)));
    let any_i64_exempt = match rust.name {
        // These intrinsics have all been manually verified against Clang's
        // headers to be available on x86, and the u64 arguments seem
        // spurious I guess?
        "_xsave" | "_xrstor" | "_xsetbv" | "_xgetbv" | "_xsaveopt" | "_xsavec" | "_xsaves"
        | "_xrstors" => true,

        // Apparently all of clang/msvc/gcc accept these intrinsics on
        // 32-bit, so let's do the same
        "_mm_set_epi64x"
        | "_mm_set1_epi64x"
        | "_mm256_set_epi64x"
        | "_mm256_setr_epi64x"
        | "_mm256_set1_epi64x"
        | "_mm512_set1_epi64"
        | "_mm256_mask_set1_epi64"
        | "_mm256_maskz_set1_epi64"
        | "_mm_mask_set1_epi64"
        | "_mm_maskz_set1_epi64"
        | "_mm512_set4_epi64"
        | "_mm512_setr4_epi64"
        | "_mm512_set_epi64"
        | "_mm512_setr_epi64"
        | "_mm512_reduce_add_epi64"
        | "_mm512_mask_reduce_add_epi64"
        | "_mm512_reduce_mul_epi64"
        | "_mm512_mask_reduce_mul_epi64"
        | "_mm512_reduce_max_epi64"
        | "_mm512_mask_reduce_max_epi64"
        | "_mm512_reduce_max_epu64"
        | "_mm512_mask_reduce_max_epu64"
        | "_mm512_reduce_min_epi64"
        | "_mm512_mask_reduce_min_epi64"
        | "_mm512_reduce_min_epu64"
        | "_mm512_mask_reduce_min_epu64"
        | "_mm512_reduce_and_epi64"
        | "_mm512_mask_reduce_and_epi64"
        | "_mm512_reduce_or_epi64"
        | "_mm512_mask_reduce_or_epi64"
        | "_mm512_mask_set1_epi64"
        | "_mm512_maskz_set1_epi64"
        | "_mm_cvt_roundss_si64"
        | "_mm_cvt_roundss_i64"
        | "_mm_cvt_roundss_u64"
        | "_mm_cvtss_i64"
        | "_mm_cvtss_u64"
        | "_mm_cvt_roundsd_si64"
        | "_mm_cvt_roundsd_i64"
        | "_mm_cvt_roundsd_u64"
        | "_mm_cvtsd_i64"
        | "_mm_cvtsd_u64"
        | "_mm_cvt_roundi64_ss"
        | "_mm_cvt_roundi64_sd"
        | "_mm_cvt_roundsi64_ss"
        | "_mm_cvt_roundsi64_sd"
        | "_mm_cvt_roundu64_ss"
        | "_mm_cvt_roundu64_sd"
        | "_mm_cvti64_ss"
        | "_mm_cvti64_sd"
        | "_mm_cvtt_roundss_si64"
        | "_mm_cvtt_roundss_i64"
        | "_mm_cvtt_roundss_u64"
        | "_mm_cvttss_i64"
        | "_mm_cvttss_u64"
        | "_mm_cvtt_roundsd_si64"
        | "_mm_cvtt_roundsd_i64"
        | "_mm_cvtt_roundsd_u64"
        | "_mm_cvttsd_i64"
        | "_mm_cvttsd_u64"
        | "_mm_cvtu64_ss"
        | "_mm_cvtu64_sd" => true,

        // These return a 64-bit argument but they're assembled from other
        // 32-bit registers, so these work on 32-bit just fine. See #308 for
        // more info.
        "_rdtsc" | "__rdtscp" => true,

        _ => false,
    };
    if any_i64 && !any_i64_exempt && !rust.file.contains("x86_64") {
        bail!(
            "intrinsic `{}` uses a 64-bit bare type but may be \
             available on 32-bit platforms",
            rust.name
        )
    }
    Ok(())
}

fn equate(t: &Type, intel: &str, intrinsic: &str, is_const: bool) -> Result<(), String> {
    // Make pointer adjacent to the type: float * foo => float* foo
    let mut intel = intel.replace(" *", "*");
    // Make mutability modifier adjacent to the pointer:
    // float const * foo => float const* foo
    intel = intel.replace("const *", "const*");
    // Normalize mutability modifier to after the type:
    // const float* foo => float const*
    if intel.starts_with("const") && intel.ends_with('*') {
        intel = intel.replace("const ", "");
        intel = intel.replace("*", " const*");
    }
    let require_const = || {
        if is_const {
            return Ok(());
        }
        Err(format!("argument required to be const but isn't"))
    };
    match (t, &intel[..]) {
        (&Type::PrimFloat(32), "float") => {}
        (&Type::PrimFloat(64), "double") => {}
        (&Type::PrimSigned(16), "__int16") => {}
        (&Type::PrimSigned(16), "short") => {}
        (&Type::PrimSigned(32), "__int32") => {}
        (&Type::PrimSigned(32), "const int") => require_const()?,
        (&Type::PrimSigned(32), "int") => {}
        (&Type::PrimSigned(64), "__int64") => {}
        (&Type::PrimSigned(64), "long long") => {}
        (&Type::PrimSigned(8), "__int8") => {}
        (&Type::PrimSigned(8), "char") => {}
        (&Type::PrimUnsigned(16), "unsigned short") => {}
        (&Type::PrimUnsigned(32), "unsigned int") => {}
        (&Type::PrimUnsigned(32), "const unsigned int") => {}
        (&Type::PrimUnsigned(64), "unsigned __int64") => {}
        (&Type::PrimUnsigned(8), "unsigned char") => {}
        (&Type::M64, "__m64") => {}
        (&Type::M128, "__m128") => {}
        (&Type::M128BH, "__m128bh") => {}
        (&Type::M128I, "__m128i") => {}
        (&Type::M128D, "__m128d") => {}
        (&Type::M256, "__m256") => {}
        (&Type::M256BH, "__m256bh") => {}
        (&Type::M256I, "__m256i") => {}
        (&Type::M256D, "__m256d") => {}
        (&Type::M512, "__m512") => {}
        (&Type::M512BH, "__m512bh") => {}
        (&Type::M512I, "__m512i") => {}
        (&Type::M512D, "__m512d") => {}
        (&Type::MMASK64, "__mmask64") => {}
        (&Type::MMASK32, "__mmask32") => {}
        (&Type::MMASK16, "__mmask16") => {}
        (&Type::MMASK8, "__mmask8") => {}

        (&Type::MutPtr(&Type::PrimFloat(32)), "float*") => {}
        (&Type::MutPtr(&Type::PrimFloat(64)), "double*") => {}
        (&Type::MutPtr(&Type::PrimFloat(32)), "void*") => {}
        (&Type::MutPtr(&Type::PrimFloat(64)), "void*") => {}
        (&Type::MutPtr(&Type::PrimSigned(32)), "void*") => {}
        (&Type::MutPtr(&Type::PrimSigned(16)), "void*") => {}
        (&Type::MutPtr(&Type::PrimSigned(8)), "void*") => {}
        (&Type::MutPtr(&Type::PrimSigned(32)), "int*") => {}
        (&Type::MutPtr(&Type::PrimSigned(32)), "__int32*") => {}
        (&Type::MutPtr(&Type::PrimSigned(64)), "void*") => {}
        (&Type::MutPtr(&Type::PrimSigned(64)), "__int64*") => {}
        (&Type::MutPtr(&Type::PrimSigned(8)), "char*") => {}
        (&Type::MutPtr(&Type::PrimUnsigned(16)), "unsigned short*") => {}
        (&Type::MutPtr(&Type::PrimUnsigned(32)), "unsigned int*") => {}
        (&Type::MutPtr(&Type::PrimUnsigned(64)), "unsigned __int64*") => {}
        (&Type::MutPtr(&Type::PrimUnsigned(8)), "void*") => {}
        (&Type::MutPtr(&Type::PrimUnsigned(32)), "__mmask32*") => {}
        (&Type::MutPtr(&Type::PrimUnsigned(64)), "__mmask64*") => {}
        (&Type::MutPtr(&Type::M64), "__m64*") => {}
        (&Type::MutPtr(&Type::M128), "__m128*") => {}
        (&Type::MutPtr(&Type::M128BH), "__m128bh*") => {}
        (&Type::MutPtr(&Type::M128I), "__m128i*") => {}
        (&Type::MutPtr(&Type::M128D), "__m128d*") => {}
        (&Type::MutPtr(&Type::M256), "__m256*") => {}
        (&Type::MutPtr(&Type::M256BH), "__m256bh*") => {}
        (&Type::MutPtr(&Type::M256I), "__m256i*") => {}
        (&Type::MutPtr(&Type::M256D), "__m256d*") => {}
        (&Type::MutPtr(&Type::M512), "__m512*") => {}
        (&Type::MutPtr(&Type::M512BH), "__m512bh*") => {}
        (&Type::MutPtr(&Type::M512I), "__m512i*") => {}
        (&Type::MutPtr(&Type::M512D), "__m512d*") => {}

        (&Type::ConstPtr(&Type::PrimFloat(32)), "float const*") => {}
        (&Type::ConstPtr(&Type::PrimFloat(64)), "double const*") => {}
        (&Type::ConstPtr(&Type::PrimFloat(32)), "void const*") => {}
        (&Type::ConstPtr(&Type::PrimFloat(64)), "void const*") => {}
        (&Type::ConstPtr(&Type::PrimSigned(32)), "int const*") => {}
        (&Type::ConstPtr(&Type::PrimSigned(32)), "__int32 const*") => {}
        (&Type::ConstPtr(&Type::PrimSigned(8)), "void const*") => {}
        (&Type::ConstPtr(&Type::PrimSigned(16)), "void const*") => {}
        (&Type::ConstPtr(&Type::PrimSigned(32)), "void const*") => {}
        (&Type::ConstPtr(&Type::PrimSigned(64)), "void const*") => {}
        (&Type::ConstPtr(&Type::PrimSigned(64)), "__int64 const*") => {}
        (&Type::ConstPtr(&Type::PrimSigned(8)), "char const*") => {}
        (&Type::ConstPtr(&Type::PrimUnsigned(16)), "unsigned short const*") => {}
        (&Type::ConstPtr(&Type::PrimUnsigned(32)), "unsigned int const*") => {}
        (&Type::ConstPtr(&Type::PrimUnsigned(64)), "unsigned __int64 const*") => {}
        (&Type::ConstPtr(&Type::PrimUnsigned(8)), "void const*") => {}
        (&Type::ConstPtr(&Type::PrimUnsigned(32)), "void const*") => {}
        (&Type::ConstPtr(&Type::M64), "__m64 const*") => {}
        (&Type::ConstPtr(&Type::M128), "__m128 const*") => {}
        (&Type::ConstPtr(&Type::M128BH), "__m128bh const*") => {}
        (&Type::ConstPtr(&Type::M128I), "__m128i const*") => {}
        (&Type::ConstPtr(&Type::M128D), "__m128d const*") => {}
        (&Type::ConstPtr(&Type::M256), "__m256 const*") => {}
        (&Type::ConstPtr(&Type::M256BH), "__m256bh const*") => {}
        (&Type::ConstPtr(&Type::M256I), "__m256i const*") => {}
        (&Type::ConstPtr(&Type::M256D), "__m256d const*") => {}
        (&Type::ConstPtr(&Type::M512), "__m512 const*") => {}
        (&Type::ConstPtr(&Type::M512BH), "__m512bh const*") => {}
        (&Type::ConstPtr(&Type::M512I), "__m512i const*") => {}
        (&Type::ConstPtr(&Type::M512D), "__m512d const*") => {}
        (&Type::ConstPtr(&Type::PrimUnsigned(32)), "__mmask32*") => {}
        (&Type::ConstPtr(&Type::PrimUnsigned(64)), "__mmask64*") => {}

        (&Type::MM_CMPINT_ENUM, "_MM_CMPINT_ENUM") => {}
        (&Type::MM_MANTISSA_NORM_ENUM, "_MM_MANTISSA_NORM_ENUM") => {}
        (&Type::MM_MANTISSA_SIGN_ENUM, "_MM_MANTISSA_SIGN_ENUM") => {}
        (&Type::MM_PERM_ENUM, "_MM_PERM_ENUM") => {}

        // This is a macro (?) in C which seems to mutate its arguments, but
        // that means that we're taking pointers to arguments in rust
        // as we're not exposing it as a macro.
        (&Type::MutPtr(&Type::M128), "__m128") if intrinsic == "_MM_TRANSPOSE4_PS" => {}

        // The _rdtsc intrinsic uses a __int64 return type, but this is a bug in
        // the intrinsics guide: https://github.com/rust-lang/stdarch/issues/559
        // We have manually fixed the bug by changing the return type to `u64`.
        (&Type::PrimUnsigned(64), "__int64") if intrinsic == "_rdtsc" => {}

        // The _bittest and _bittest64 intrinsics takes a mutable pointer in the
        // intrinsics guide even though it never writes through the pointer:
        (&Type::ConstPtr(&Type::PrimSigned(32)), "__int32*") if intrinsic == "_bittest" => {}
        (&Type::ConstPtr(&Type::PrimSigned(64)), "__int64*") if intrinsic == "_bittest64" => {}
        // The _xrstor, _fxrstor, _xrstor64, _fxrstor64 intrinsics take a
        // mutable pointer in the intrinsics guide even though they never write
        // through the pointer:
        (&Type::ConstPtr(&Type::PrimUnsigned(8)), "void*")
            if intrinsic == "_xrstor"
                || intrinsic == "_xrstor64"
                || intrinsic == "_fxrstor"
                || intrinsic == "_fxrstor64" => {}

        _ => bail!(
            "failed to equate: `{}` and {:?} for {}",
            intel,
            t,
            intrinsic
        ),
    }
    Ok(())
}
