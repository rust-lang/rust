#![allow(bad_style)]
#![cfg_attr(
    feature = "cargo-clippy",
    allow(
        clippy::shadow_reuse,
        clippy::cast_lossless,
        clippy::match_same_arms,
        clippy::nonminimal_bool,
        clippy::print_stdout,
        clippy::use_debug,
        clippy::eq_op,
        clippy::useless_format
    )
)]

#[macro_use]
extern crate serde_derive;
extern crate serde_xml_rs;
extern crate stdsimd_verify;

use std::collections::{BTreeMap, HashMap};

use stdsimd_verify::x86_functions;

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
}

static F32: Type = Type::PrimFloat(32);
static F64: Type = Type::PrimFloat(64);
static I16: Type = Type::PrimSigned(16);
static I32: Type = Type::PrimSigned(32);
static I64: Type = Type::PrimSigned(64);
static I8: Type = Type::PrimSigned(8);
static U16: Type = Type::PrimUnsigned(16);
static U32: Type = Type::PrimUnsigned(32);
static U64: Type = Type::PrimUnsigned(64);
static U8: Type = Type::PrimUnsigned(8);

static M64: Type = Type::M64;
static M128: Type = Type::M128;
static M128I: Type = Type::M128I;
static M128D: Type = Type::M128D;
static M256: Type = Type::M256;
static M256I: Type = Type::M256I;
static M256D: Type = Type::M256D;
static M512: Type = Type::M512;
static M512I: Type = Type::M512I;
static M512D: Type = Type::M512D;
static MMASK16: Type = Type::MMASK16;

static TUPLE: Type = Type::Tuple;
static CPUID: Type = Type::CpuidResult;
static NEVER: Type = Type::Never;

#[derive(Debug)]
enum Type {
    PrimFloat(u8),
    PrimSigned(u8),
    PrimUnsigned(u8),
    Ptr(&'static Type),
    M64,
    M128,
    M128D,
    M128I,
    M256,
    M256D,
    M256I,
    M512,
    M512D,
    M512I,
    MMASK16,
    Tuple,
    CpuidResult,
    Never,
}

x86_functions!(static FUNCTIONS);

#[derive(Deserialize)]
struct Data {
    #[serde(rename = "intrinsic", default)]
    intrinsics: Vec<Intrinsic>,
}

#[derive(Deserialize)]
struct Intrinsic {
    rettype: String,
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
    let data: Data = serde_xml_rs::deserialize(xml).expect("failed to deserialize xml");
    let mut map = HashMap::new();
    for intrinsic in &data.intrinsics {
        map.entry(&intrinsic.name[..])
            .or_insert_with(Vec::new)
            .push(intrinsic);
    }

    let mut all_valid = true;
    'outer: for rust in FUNCTIONS {
        match rust.name {
            // These aren't defined by Intel but they're defined by what appears
            // to be all other compilers. For more information see
            // rust-lang-nursery/stdsimd#307, and otherwise these signatures
            // have all been manually verified.
            "__readeflags" |
            "__writeeflags" |
            "__cpuid_count" |
            "__cpuid" |
            "__get_cpuid_max" |
            // The UD2 intrinsic is not defined by Intel, but it was agreed on
            // in the RFC Issue 2512:
            // https://github.com/rust-lang/rfcs/issues/2512
            "ud2"
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
        "_bswap" => {}
        "_bswap64" => {}
        _ => {
            if intel.cpuid.is_empty() {
                bail!("missing cpuid for {}", rust.name);
            }
        }
    }

    for cpuid in &intel.cpuid {
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

        let rust_feature = rust
            .target_feature
            .expect(&format!("no target feature listed for {}", rust.name));
        if rust_feature.contains(&cpuid) {
            continue;
        }
        bail!(
            "intel cpuid `{}` not in `{}` for {}",
            cpuid,
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
        equate(t, &intel.rettype, rust.name, false)?;
    } else if intel.rettype != "" && intel.rettype != "void" {
        bail!(
            "{} returns `{}` with intel, void in rust",
            rust.name,
            intel.rettype
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
        .any(|arg| match *arg {
            Type::PrimSigned(64) | Type::PrimUnsigned(64) => true,
            _ => false,
        });
    let any_i64_exempt = match rust.name {
        // These intrinsics have all been manually verified against Clang's
        // headers to be available on x86, and the u64 arguments seem
        // spurious I guess?
        "_xsave" | "_xrstor" | "_xsetbv" | "_xgetbv" | "_xsaveopt" | "_xsavec" | "_xsaves"
        | "_xrstors" => true,

        // Apparently all of clang/msvc/gcc accept these intrinsics on
        // 32-bit, so let's do the same
        "_mm_set_epi64x" | "_mm_set1_epi64x" | "_mm256_set_epi64x" | "_mm256_setr_epi64x"
        | "_mm256_set1_epi64x" => true,

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
    let intel = intel.replace(" *", "*");
    let intel = intel.replace(" const*", "*");
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
        (&Type::PrimUnsigned(64), "unsigned __int64") => {}
        (&Type::PrimUnsigned(8), "unsigned char") => {}

        (&Type::Ptr(&Type::PrimFloat(32)), "float*") => {}
        (&Type::Ptr(&Type::PrimFloat(64)), "double*") => {}
        (&Type::Ptr(&Type::PrimSigned(32)), "int*") => {}
        (&Type::Ptr(&Type::PrimSigned(64)), "__int64*") => {}
        (&Type::Ptr(&Type::PrimSigned(8)), "char*") => {}
        (&Type::Ptr(&Type::PrimUnsigned(16)), "unsigned short*") => {}
        (&Type::Ptr(&Type::PrimUnsigned(32)), "unsigned int*") => {}
        (&Type::Ptr(&Type::PrimUnsigned(64)), "unsigned __int64*") => {}
        (&Type::Ptr(&Type::PrimUnsigned(8)), "const void*") => {}
        (&Type::Ptr(&Type::PrimUnsigned(8)), "void*") => {}

        (&Type::M64, "__m64") | (&Type::Ptr(&Type::M64), "__m64*") => {}

        (&Type::M128I, "__m128i")
        | (&Type::Ptr(&Type::M128I), "__m128i*")
        | (&Type::M128D, "__m128d")
        | (&Type::Ptr(&Type::M128D), "__m128d*")
        | (&Type::M128, "__m128")
        | (&Type::Ptr(&Type::M128), "__m128*") => {}

        (&Type::M256I, "__m256i")
        | (&Type::Ptr(&Type::M256I), "__m256i*")
        | (&Type::M256D, "__m256d")
        | (&Type::Ptr(&Type::M256D), "__m256d*")
        | (&Type::M256, "__m256")
        | (&Type::Ptr(&Type::M256), "__m256*") => {}

        (&Type::M512I, "__m512i")
        | (&Type::Ptr(&Type::M512I), "__m512i*")
        | (&Type::M512D, "__m512d")
        | (&Type::Ptr(&Type::M512D), "__m512d*")
        | (&Type::M512, "__m512")
        | (&Type::Ptr(&Type::M512), "__m512*") => {}

        (&Type::MMASK16, "__mmask16") => {}

        // This is a macro (?) in C which seems to mutate its arguments, but
        // that means that we're taking pointers to arguments in rust
        // as we're not exposing it as a macro.
        (&Type::Ptr(&Type::M128), "__m128") if intrinsic == "_MM_TRANSPOSE4_PS" => {}

        _ => bail!(
            "failed to equate: `{}` and {:?} for {}",
            intel,
            t,
            intrinsic
        ),
    }
    Ok(())
}
