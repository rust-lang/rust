#![allow(unused, non_camel_case_types)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io;
use std::io::{BufWriter, Write};

use serde::Deserialize;

const PRINT_INSTRUCTION_VIOLATIONS: bool = false;
const GENERATE_MISSING_X86_MD: bool = false;
const SS: u8 = (8 * size_of::<usize>()) as u8;

struct Function {
    name: &'static str,
    arguments: &'static [&'static Type],
    ret: Option<&'static Type>,
    target_feature: Option<&'static str>,
    instrs: &'static [&'static str],
    file: &'static str,
    required_const: &'static [usize],
    has_test: bool,
    doc: &'static str,
}

static BF16: Type = Type::BFloat16;
static F16: Type = Type::PrimFloat(16);
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
static USIZE: Type = Type::PrimUnsigned(SS);
static ORDERING: Type = Type::Ordering;

static M128: Type = Type::M128;
static M128BH: Type = Type::M128BH;
static M128I: Type = Type::M128I;
static M128D: Type = Type::M128D;
static M128H: Type = Type::M128H;
static M256: Type = Type::M256;
static M256BH: Type = Type::M256BH;
static M256I: Type = Type::M256I;
static M256D: Type = Type::M256D;
static M256H: Type = Type::M256H;
static M512: Type = Type::M512;
static M512BH: Type = Type::M512BH;
static M512I: Type = Type::M512I;
static M512D: Type = Type::M512D;
static M512H: Type = Type::M512H;
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

#[derive(Debug, PartialEq, Copy, Clone)]
enum Type {
    PrimFloat(u8),
    PrimSigned(u8),
    PrimUnsigned(u8),
    BFloat16,
    MutPtr(&'static Type),
    ConstPtr(&'static Type),
    M128,
    M128BH,
    M128D,
    M128H,
    M128I,
    M256,
    M256BH,
    M256D,
    M256H,
    M256I,
    M512,
    M512BH,
    M512D,
    M512H,
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
    #[serde(rename = "@name")]
    name: String,
    #[serde(rename = "@tech")]
    tech: String,
    #[serde(rename = "CPUID", default)]
    cpuid: Vec<String>,
    #[serde(rename = "parameter", default)]
    parameters: Vec<Parameter>,
    #[serde(rename = "@sequence", default)]
    generates_sequence: bool,
    #[serde(default)]
    instruction: Vec<Instruction>,
}

#[derive(Deserialize)]
struct Parameter {
    #[serde(rename = "@type")]
    type_: String,
    #[serde(rename = "@etype", default)]
    etype: String,
}

#[derive(Deserialize)]
struct Return {
    #[serde(rename = "@type", default)]
    type_: String,
}

#[derive(Deserialize, Debug)]
struct Instruction {
    #[serde(rename = "@name")]
    name: String,
}

macro_rules! bail {
    ($($t:tt)*) => { return Err(format!($($t)*)) }
}

#[test]
fn verify_all_signatures() {
    // This XML document was downloaded from Intel's site. To update this you
    // can visit intel's intrinsics guide online documentation:
    //
    //   https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
    //
    // Open up the network console and you'll see an xml file was downloaded
    // (currently called data-3.6.9.xml). That's the file we downloaded
    // here.
    let xml = include_bytes!("../x86-intel.xml");

    let xml = &xml[..];
    let data: Data = quick_xml::de::from_reader(xml).expect("failed to deserialize xml");
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
                // MXCSR - deprecated, immediate UB
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
                // CPUID
                "__cpuid_count",
                "__cpuid",
                "__get_cpuid_max",
                // Privileged, see https://github.com/rust-lang/stdarch/issues/209
                "_xsetbv",
                "_xsaves",
                "_xrstors",
                "_xsaves64",
                "_xrstors64",
                "_mm_loadiwkey",
                // RDRAND
                "_rdrand16_step",
                "_rdrand32_step",
                "_rdrand64_step",
                "_rdseed16_step",
                "_rdseed32_step",
                "_rdseed64_step",
                // Prefetch
                "_mm_prefetch",
                // CMPXCHG
                "cmpxchg16b",
                // Undefined
                "_mm_undefined_ps",
                "_mm_undefined_pd",
                "_mm_undefined_si128",
                "_mm_undefined_ph",
                "_mm256_undefined_ps",
                "_mm256_undefined_pd",
                "_mm256_undefined_si256",
                "_mm256_undefined_ph",
                "_mm512_undefined_ps",
                "_mm512_undefined_pd",
                "_mm512_undefined_epi32",
                "_mm512_undefined",
                "_mm512_undefined_ph",
                // Has doc-tests instead
                "_mm256_shuffle_epi32",
                "_mm256_unpackhi_epi8",
                "_mm256_unpacklo_epi8",
                "_mm256_unpackhi_epi16",
                "_mm256_unpacklo_epi16",
                "_mm256_unpackhi_epi32",
                "_mm256_unpacklo_epi32",
                "_mm256_unpackhi_epi64",
                "_mm256_unpacklo_epi64",
                // Has tests with some other intrinsic
                "__writeeflags",
                "_xrstor",
                "_xrstor64",
                "_fxrstor",
                "_fxrstor64",
                "_xend",
                "_xabort_code",
                // Aliases
                "_mm_comige_ss",
                "_mm_cvt_ss2si",
                "_mm_cvtt_ss2si",
                "_mm_cvt_si2ss",
                "_mm_set_ps1",
                "_mm_load_ps1",
                "_mm_store_ps1",
                "_mm_bslli_si128",
                "_mm_bsrli_si128",
                "_bextr2_u32",
                "_mm_tzcnt_32",
                "_mm256_bslli_epi128",
                "_mm256_bsrli_epi128",
                "_mm_cvtsi64x_si128",
                "_mm_cvtsi128_si64x",
                "_mm_cvtsi64x_sd",
                "_bextr2_u64",
                "_mm_tzcnt_64",
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
            "_MM_SHUFFLE" |
            "_xabort_code" |
            // Not listed with intel, but manually verified
            "cmpxchg16b"
            => continue,
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
            println!("  * {error}");
        }
        all_valid = false;
    }
    assert!(all_valid);

    if GENERATE_MISSING_X86_MD {
        print_missing(
            &map,
            BufWriter::new(File::create("../core_arch/missing-x86.md").unwrap()),
        )
        .unwrap();
    }
}

fn print_missing(map: &HashMap<&str, Vec<&Intrinsic>>, mut f: impl Write) -> io::Result<()> {
    let mut missing = BTreeMap::new(); // BTreeMap to keep the cpuids ordered

    // we cannot use SVML and MMX, and MPX is not in LLVM, and intrinsics without any cpuid requirement
    // are accessible from safe rust
    for intrinsic in map.values().flatten().filter(|intrinsic| {
        intrinsic.tech != "SVML"
            && intrinsic.tech != "MMX"
            && !intrinsic.cpuid.is_empty()
            && !intrinsic.cpuid.contains(&"MPX".to_string())
            && intrinsic.return_.type_ != "__m64"
            && !intrinsic
                .parameters
                .iter()
                .any(|param| param.type_.contains("__m64"))
    }) {
        missing
            .entry(&intrinsic.cpuid)
            .or_insert_with(Vec::new)
            .push(intrinsic);
    }

    for (k, v) in &mut missing {
        v.sort_by_key(|intrinsic| &intrinsic.name); // sort to make the order of everything same
        writeln!(f, "\n<details><summary>{k:?}</summary><p>\n")?;
        for intel in v {
            let url = format!(
                "https://software.intel.com/sites/landingpage\
                         /IntrinsicsGuide/#text={}",
                intel.name
            );
            writeln!(f, "  * [ ] [`{}`]({url})", intel.name)?;
        }
        writeln!(f, "</p></details>\n")?;
    }

    f.flush()
}

fn check_target_features(rust: &Function, intel: &Intrinsic) -> Result<(), String> {
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

    let rust_features = match rust.target_feature {
        Some(features) => features
            .split(',')
            .map(|feature| feature.to_string())
            .collect(),
        None => HashSet::new(),
    };

    let mut intel_cpuids = HashSet::new();

    for cpuid in &intel.cpuid {
        // The pause intrinsic is in the SSE2 module, but it is backwards
        // compatible with CPUs without SSE2, and it therefore does not need the
        // target-feature attribute.
        if rust.name == "_mm_pause" {
            continue;
        }

        // these flags on the rdtsc/rtdscp intrinsics we don't test for right
        // now, but we may wish to add these one day!
        //
        // For more info see #308
        if *cpuid == "TSC" || *cpuid == "RDTSCP" {
            continue;
        }

        // Some CPUs support VAES/GFNI/VPCLMULQDQ without AVX512, even though
        // the Intel documentation states that those instructions require
        // AVX512VL.
        if *cpuid == "AVX512VL"
            && intel
                .cpuid
                .iter()
                .any(|x| matches!(&**x, "VAES" | "GFNI" | "VPCLMULQDQ"))
        {
            continue;
        }

        let cpuid = cpuid.to_lowercase().replace('_', "");

        // Fix mismatching feature names:
        let fixed_cpuid = match cpuid.as_ref() {
            // The XML file names IFMA as "avx512ifma52", while Rust calls
            // it "avx512ifma".
            "avx512ifma52" => String::from("avx512ifma"),
            "xss" => String::from("xsaves"),
            "keylocker" => String::from("kl"),
            "keylockerwide" => String::from("widekl"),
            _ => cpuid,
        };

        intel_cpuids.insert(fixed_cpuid);
    }

    if intel_cpuids.contains("gfni") {
        if rust.name.contains("mask") {
            // LLVM requires avx512bw for all masked GFNI intrinsics, and also avx512vl for the 128- and 256-bit versions
            if !rust.name.starts_with("_mm512") {
                intel_cpuids.insert(String::from("avx512vl"));
            }
            intel_cpuids.insert(String::from("avx512bw"));
        } else if rust.name.starts_with("_mm256") {
            // LLVM requires AVX for all non-masked 256-bit GFNI intrinsics
            intel_cpuids.insert(String::from("avx"));
        }
    }

    // Also, 512-bit vpclmulqdq intrisic requires avx512f
    if &rust.name == &"_mm512_clmulepi64_epi128" {
        intel_cpuids.insert(String::from("avx512f"));
    }

    if rust_features != intel_cpuids {
        bail!(
            "Intel cpuids `{:?}` doesn't match Rust `{:?}` for {}",
            intel_cpuids,
            rust_features,
            rust.name
        );
    }

    Ok(())
}

fn matches(rust: &Function, intel: &Intrinsic) -> Result<(), String> {
    check_target_features(rust, intel)?;

    if PRINT_INSTRUCTION_VIOLATIONS {
        if rust.instrs.is_empty() {
            if !intel.instruction.is_empty() && !intel.generates_sequence {
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
                let asserting = intel
                    .instruction
                    .iter()
                    .any(|a| a.name.to_lowercase().starts_with(instr));
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
        equate(t, &intel.return_.type_, "", intel, false)?;
    } else if !intel.return_.type_.is_empty() && intel.return_.type_ != "void" {
        bail!(
            "{} returns `{}` with intel, void in rust",
            rust.name,
            intel.return_.type_
        );
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
            bail!("wrong number of arguments on {}", rust.name);
        }
        for (i, (a, b)) in intel.parameters.iter().zip(rust.arguments).enumerate() {
            let is_const = rust.required_const.contains(&i);
            equate(b, &a.type_, &a.etype, &intel, is_const)?;
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
        );
    }
    if !rust.doc.contains("Intel") {
        bail!("No link to Intel");
    }
    let recognized_links = [
        "https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html",
        "https://software.intel.com/sites/landingpage/IntrinsicsGuide/",
    ];
    if !recognized_links.iter().any(|link| rust.doc.contains(link)) {
        bail!("Unrecognized Intel Link");
    }
    if !rust.doc.contains(&rust.name[1..]) {
        // We can leave the leading underscore
        bail!("Bad link to Intel");
    }
    Ok(())
}

fn pointed_type(intrinsic: &Intrinsic) -> Result<Type, String> {
    Ok(
        if intrinsic.tech == "AMX"
            || intrinsic
                .cpuid
                .iter()
                .any(|cpuid| matches!(&**cpuid, "KEYLOCKER" | "KEYLOCKER_WIDE" | "XSAVE" | "FXSR"))
        {
            // AMX, KEYLOCKER and XSAVE intrinsics should take `*u8`
            U8
        } else if intrinsic.name == "_mm_clflush" {
            // Just a false match in the following logic
            U8
        } else if ["_mm_storeu_si", "_mm_loadu_si"]
            .iter()
            .any(|x| intrinsic.name.starts_with(x))
        {
            // These have already been stabilized, so cannot be changed anymore
            U8
        } else if intrinsic.name.ends_with("i8") {
            I8
        } else if intrinsic.name.ends_with("i16") {
            I16
        } else if intrinsic.name.ends_with("i32") {
            I32
        } else if intrinsic.name.ends_with("i64") {
            I64
        } else if intrinsic.name.ends_with("i128") {
            M128I
        } else if intrinsic.name.ends_with("i256") {
            M256I
        } else if intrinsic.name.ends_with("i512") {
            M512I
        } else if intrinsic.name.ends_with("h") {
            F16
        } else if intrinsic.name.ends_with("s") {
            F32
        } else if intrinsic.name.ends_with("d") {
            F64
        } else {
            bail!(
                "Don't know what type of *void to use for {}",
                intrinsic.name
            );
        },
    )
}

fn equate(
    t: &Type,
    intel: &str,
    etype: &str,
    intrinsic: &Intrinsic,
    is_const: bool,
) -> Result<(), String> {
    // Make pointer adjacent to the type: float * foo => float* foo
    let mut intel = intel.replace(" *", "*");
    // Make mutability modifier adjacent to the pointer:
    // float const * foo => float const* foo
    intel = intel.replace("const *", "const*");
    // Normalize mutability modifier to after the type:
    // const float* foo => float const*
    if intel.starts_with("const") && intel.ends_with('*') {
        intel = intel.replace("const ", "");
        intel = intel.replace('*', " const*");
    }
    if etype == "IMM" || intel == "constexpr int" {
        // The _bittest intrinsics claim to only accept immediates but actually
        // accept run-time values as well.
        if !is_const && !intrinsic.name.starts_with("_bittest") {
            bail!("argument required to be const but isn't");
        }
    } else {
        // const int must be an IMM
        assert_ne!(intel, "const int");
        if is_const {
            bail!("argument is const but shouldn't be");
        }
    }
    match (t, &intel[..]) {
        (&Type::PrimFloat(16), "_Float16") => {}
        (&Type::PrimFloat(32), "float") => {}
        (&Type::PrimFloat(64), "double") => {}
        (&Type::PrimSigned(8), "__int8" | "char") => {}
        (&Type::PrimSigned(16), "__int16" | "short") => {}
        (&Type::PrimSigned(32), "__int32" | "constexpr int" | "const int" | "int") => {}
        (&Type::PrimSigned(64), "__int64" | "long long") => {}
        (&Type::PrimUnsigned(8), "unsigned char") => {}
        (&Type::PrimUnsigned(16), "unsigned short") => {}
        (&Type::BFloat16, "__bfloat16") => {}
        (
            &Type::PrimUnsigned(32),
            "unsigned __int32" | "unsigned int" | "unsigned long" | "const unsigned int",
        ) => {}
        (&Type::PrimUnsigned(64), "unsigned __int64") => {}
        (&Type::PrimUnsigned(SS), "size_t") => {}

        (&Type::M128, "__m128") => {}
        (&Type::M128BH, "__m128bh") => {}
        (&Type::M128I, "__m128i") => {}
        (&Type::M128D, "__m128d") => {}
        (&Type::M128H, "__m128h") => {}
        (&Type::M256, "__m256") => {}
        (&Type::M256BH, "__m256bh") => {}
        (&Type::M256I, "__m256i") => {}
        (&Type::M256D, "__m256d") => {}
        (&Type::M256H, "__m256h") => {}
        (&Type::M512, "__m512") => {}
        (&Type::M512BH, "__m512bh") => {}
        (&Type::M512I, "__m512i") => {}
        (&Type::M512D, "__m512d") => {}
        (&Type::M512H, "__m512h") => {}
        (&Type::MMASK64, "__mmask64") => {}
        (&Type::MMASK32, "__mmask32") => {}
        (&Type::MMASK16, "__mmask16") => {}
        (&Type::MMASK8, "__mmask8") => {}

        (&Type::MutPtr(_type), "void*") | (&Type::ConstPtr(_type), "void const*") => {
            let pointed_type = pointed_type(intrinsic)?;
            if _type != &pointed_type {
                bail!(
                    "incorrect void pointer type {_type:?} in {}, should be pointer to {pointed_type:?}",
                    intrinsic.name,
                );
            }
        }

        (&Type::MutPtr(&Type::PrimFloat(32)), "float*") => {}
        (&Type::MutPtr(&Type::PrimFloat(64)), "double*") => {}
        (&Type::MutPtr(&Type::PrimSigned(8)), "char*") => {}
        (&Type::MutPtr(&Type::PrimSigned(32)), "__int32*" | "int*") => {}
        (&Type::MutPtr(&Type::PrimSigned(64)), "__int64*") => {}
        (&Type::MutPtr(&Type::PrimUnsigned(8)), "unsigned char*") => {}
        (&Type::MutPtr(&Type::PrimUnsigned(16)), "unsigned short*") => {}
        (&Type::MutPtr(&Type::PrimUnsigned(32)), "unsigned int*" | "unsigned __int32*") => {}
        (&Type::MutPtr(&Type::PrimUnsigned(64)), "unsigned __int64*") => {}

        (&Type::MutPtr(&Type::MMASK8), "__mmask8*") => {}
        (&Type::MutPtr(&Type::MMASK32), "__mmask32*") => {}
        (&Type::MutPtr(&Type::MMASK64), "__mmask64*") => {}
        (&Type::MutPtr(&Type::MMASK16), "__mmask16*") => {}

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

        (&Type::ConstPtr(&Type::PrimFloat(16)), "_Float16 const*") => {}
        (&Type::ConstPtr(&Type::PrimFloat(32)), "float const*") => {}
        (&Type::ConstPtr(&Type::PrimFloat(64)), "double const*") => {}
        (&Type::ConstPtr(&Type::PrimSigned(8)), "char const*") => {}
        (&Type::ConstPtr(&Type::PrimSigned(32)), "__int32 const*" | "int const*") => {}
        (&Type::ConstPtr(&Type::PrimSigned(64)), "__int64 const*") => {}
        (&Type::ConstPtr(&Type::PrimUnsigned(16)), "unsigned short const*") => {}
        (&Type::ConstPtr(&Type::PrimUnsigned(32)), "unsigned int const*") => {}
        (&Type::ConstPtr(&Type::PrimUnsigned(64)), "unsigned __int64 const*") => {}
        (&Type::ConstPtr(&Type::BFloat16), "__bf16 const*") => {}

        (&Type::ConstPtr(&Type::M128), "__m128 const*") => {}
        (&Type::ConstPtr(&Type::M128BH), "__m128bh const*") => {}
        (&Type::ConstPtr(&Type::M128I), "__m128i const*") => {}
        (&Type::ConstPtr(&Type::M128D), "__m128d const*") => {}
        (&Type::ConstPtr(&Type::M128H), "__m128h const*") => {}
        (&Type::ConstPtr(&Type::M256), "__m256 const*") => {}
        (&Type::ConstPtr(&Type::M256BH), "__m256bh const*") => {}
        (&Type::ConstPtr(&Type::M256I), "__m256i const*") => {}
        (&Type::ConstPtr(&Type::M256D), "__m256d const*") => {}
        (&Type::ConstPtr(&Type::M256H), "__m256h const*") => {}
        (&Type::ConstPtr(&Type::M512), "__m512 const*") => {}
        (&Type::ConstPtr(&Type::M512BH), "__m512bh const*") => {}
        (&Type::ConstPtr(&Type::M512I), "__m512i const*") => {}
        (&Type::ConstPtr(&Type::M512D), "__m512d const*") => {}

        (&Type::ConstPtr(&Type::MMASK8), "__mmask8*") => {}
        (&Type::ConstPtr(&Type::MMASK16), "__mmask16*") => {}
        (&Type::ConstPtr(&Type::MMASK32), "__mmask32*") => {}
        (&Type::ConstPtr(&Type::MMASK64), "__mmask64*") => {}

        (&Type::MM_CMPINT_ENUM, "_MM_CMPINT_ENUM") => {}
        (&Type::MM_MANTISSA_NORM_ENUM, "_MM_MANTISSA_NORM_ENUM") => {}
        (&Type::MM_MANTISSA_SIGN_ENUM, "_MM_MANTISSA_SIGN_ENUM") => {}
        (&Type::MM_PERM_ENUM, "_MM_PERM_ENUM") => {}

        // This is a macro (?) in C which seems to mutate its arguments, but
        // that means that we're taking pointers to arguments in rust
        // as we're not exposing it as a macro.
        (&Type::MutPtr(&Type::M128), "__m128") if intrinsic.name == "_MM_TRANSPOSE4_PS" => {}

        // The _rdtsc intrinsic uses a __int64 return type, but this is a bug in
        // the intrinsics guide: https://github.com/rust-lang/stdarch/issues/559
        // We have manually fixed the bug by changing the return type to `u64`.
        (&Type::PrimUnsigned(64), "__int64") if intrinsic.name == "_rdtsc" => {}

        // The _bittest and _bittest64 intrinsics takes a mutable pointer in the
        // intrinsics guide even though it never writes through the pointer:
        (&Type::ConstPtr(&Type::PrimSigned(32)), "__int32*") if intrinsic.name == "_bittest" => {}
        (&Type::ConstPtr(&Type::PrimSigned(64)), "__int64*") if intrinsic.name == "_bittest64" => {}
        // The _xrstor, _fxrstor, _xrstor64, _fxrstor64 intrinsics take a
        // mutable pointer in the intrinsics guide even though they never write
        // through the pointer:
        (&Type::ConstPtr(&Type::PrimUnsigned(8)), "void*")
            if matches!(
                &*intrinsic.name,
                "_xrstor" | "_xrstor64" | "_fxrstor" | "_fxrstor64"
            ) => {}
        // The _mm_stream_load_si128 intrinsic take a mutable pointer in the intrinsics
        // guide even though they never write through the pointer
        (&Type::ConstPtr(&Type::M128I), "void*") if intrinsic.name == "_mm_stream_load_si128" => {}
        /// Intel requires the mask argument for _mm_shuffle_ps to be an
        // unsigned integer, but all other _mm_shuffle_.. intrinsics
        // take a signed-integer. This breaks `_MM_SHUFFLE` for
        // `_mm_shuffle_ps`
        (&Type::PrimSigned(32), "unsigned int") if intrinsic.name == "_mm_shuffle_ps" => {}

        _ => bail!(
            "failed to equate: `{intel}` and {t:?} for {}",
            intrinsic.name
        ),
    }
    Ok(())
}
