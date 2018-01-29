#![feature(proc_macro)]
#![allow(bad_style)]
#![cfg_attr(feature = "cargo-clippy",
            allow(shadow_reuse, cast_lossless, match_same_arms))]

#[macro_use]
extern crate serde_derive;
extern crate serde_xml_rs;
extern crate stdsimd_verify;

use std::collections::HashMap;

use stdsimd_verify::x86_functions;

struct Function {
    name: &'static str,
    arguments: &'static [&'static Type],
    ret: Option<&'static Type>,
    target_feature: Option<&'static str>,
    instrs: &'static [&'static str],
    file: &'static str,
}

static BOOL: Type = Type::Bool;
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

static TUPLE: Type = Type::Tuple;
static CPUID: Type = Type::CpuidResult;

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
    Bool,
    Tuple,
    CpuidResult,
}

x86_functions!(static FUNCTIONS);

#[derive(Deserialize)]
struct Data {
    #[serde(rename = "intrinsic", default)] intrinsics: Vec<Intrinsic>,
}

#[derive(Deserialize)]
struct Intrinsic {
    rettype: String,
    name: String,
    tech: String,
    #[serde(rename = "CPUID", default)] cpuid: Vec<String>,
    #[serde(rename = "parameter", default)] parameters: Vec<Parameter>,
    #[serde(default)] instruction: Vec<Instruction>,
}

#[derive(Deserialize)]
struct Parameter {
    #[serde(rename = "type")] type_: String,
}

#[derive(Deserialize, Debug)]
struct Instruction {
    name: String,
}

fn skip_intrinsic(name: &str) -> bool {
    match name {
        // This intrinsic has multiple definitions in the XML, so just
        // ignore it.
        "_mm_prefetch" => true,

        // FIXME(#307)
        "__readeflags" |
        "__writeeflags" => true,
        "__cpuid_count" => true,
        "__cpuid" => true,
        "__get_cpuid_max" => true,

        _ => false,
    }
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
    let data: Data =
        serde_xml_rs::deserialize(xml).expect("failed to deserialize xml");
    let mut map = HashMap::new();
    for intrinsic in &data.intrinsics {
        if skip_intrinsic(&intrinsic.name) {
            continue
        }

        // These'll need to get added eventually, but right now they have some
        // duplicate names in the XML which we're not dealing with yet
        if intrinsic.tech == "AVX-512" {
            continue;
        }

        assert!(map.insert(&intrinsic.name[..], intrinsic).is_none());
    }

    for rust in FUNCTIONS {
        if skip_intrinsic(&rust.name) {
            continue;
        }

        // these are all AMD-specific intrinsics
        if let Some(feature) = rust.target_feature {
            if feature.contains("sse4a") || feature.contains("tbm") {
                continue;
            }
        }

        let intel = match map.get(rust.name) {
            Some(i) => i,
            None => panic!("missing intel definition for {}", rust.name),
        };

        // Verify that all `#[target_feature]` annotations are correct,
        // ensuring that we've actually enabled the right instruction
        // set for this intrinsic.
        match rust.name {
            "_bswap" => {}
            "_bswap64" => {}
            _ => {
                assert!(!intel.cpuid.is_empty(), "missing cpuid for {}", rust.name);
            }
        }
        for cpuid in &intel.cpuid {
            // this is needed by _xsave and probably some related intrinsics,
            // but let's just skip it for now.
            if *cpuid == "XSS" {
                continue;
            }

            // FIXME(#308)
            if *cpuid == "TSC" || *cpuid == "RDTSCP" {
                continue;
            }

            let cpuid = cpuid
                .chars()
                .flat_map(|c| c.to_lowercase())
                .collect::<String>();

            // Normalize `bmi1` to `bmi` as apparently that's what we're
            // calling it.
            let cpuid = if cpuid == "bmi1" {
                String::from("bmi")
            } else {
                cpuid
            };

            let rust_feature = rust.target_feature
                .expect(&format!("no target feature listed for {}", rust.name));
            assert!(
                rust_feature.contains(&cpuid),
                "intel cpuid `{}` not in `{}` for {}",
                cpuid,
                rust_feature,
                rust.name
            );
        }

        if rust.instrs.is_empty() {
            if intel.instruction.len() > 0 {
                println!("instruction not listed for `{}`, but intel lists {:?}",
                         rust.name, intel.instruction);
            }

        // If intel doesn't list any instructions and we do then don't
        // bother trying to look for instructions in intel, we've just got
        // some extra assertions on our end.
        } else if !intel.instruction.is_empty() {
            for instr in rust.instrs {
                let asserting = intel
                    .instruction
                    .iter()
                    .any(|a| a.name.starts_with(instr));
                if !asserting {
                    println!(
                        "intel failed to list `{}` as an instruction for `{}`",
                        instr,
                        rust.name
                    );
                }
            }
        }

        // Make sure we've got the right return type.
        if let Some(t) = rust.ret {
            equate(t, &intel.rettype, rust.name);
        } else {
            assert!(
                intel.rettype == "" || intel.rettype == "void",
                "{} returns `{}` with intel, void in rust",
                rust.name,
                intel.rettype
            );
        }

        // If there's no arguments on Rust's side intel may list one "void"
        // argument, so handle that here.
        if rust.arguments.is_empty() && intel.parameters.len() == 1 {
            assert_eq!(intel.parameters[0].type_, "void");
        } else {
            // Otherwise we want all parameters to be exactly the same
            assert_eq!(
                rust.arguments.len(),
                intel.parameters.len(),
                "wrong number of arguments on {}",
                rust.name
            );
            for (a, b) in intel.parameters.iter().zip(rust.arguments) {
                equate(b, &a.type_, &intel.name);
            }
        }

        let any_i64 = rust.arguments.iter()
            .cloned()
            .chain(rust.ret)
            .any(|arg| {
                match *arg {
                    Type::PrimSigned(64) |
                    Type::PrimUnsigned(64) => true,
                    _ => false,
                }
            });
        let any_i64_exempt = match rust.name {
            // These intrinsics have all been manually verified against Clang's
            // headers to be available on x86, and the u64 arguments seem
            // spurious I guess?
            "_xsave" |
            "_xrstor" |
            "_xsetbv" |
            "_xgetbv" |
            "_xsaveopt" |
            "_xsavec" |
            "_xsaves" |
            "_xrstors" => true,

            // Apparently all of clang/msvc/gcc accept these intrinsics on
            // 32-bit, so let's do the same
            "_mm_set_epi64x" |
            "_mm_set1_epi64x" |
            "_mm256_set_epi64x" |
            "_mm256_setr_epi64x" |
            "_mm256_set1_epi64x" => true,

            // FIXME(#308)
            "_rdtsc" |
            "__rdtscp" => true,

            _ => false,
        };
        if any_i64 && !any_i64_exempt {
            assert!(rust.file.contains("x86_64"),
                    "intrinsic `{}` uses a 64-bit bare type but may be \
                     available on 32-bit platforms",
                    rust.name);
        }
    }
}

fn equate(t: &Type, intel: &str, intrinsic: &str) {
    let intel = intel.replace(" *", "*");
    let intel = intel.replace(" const*", "*");
    match (t, &intel[..]) {
        (&Type::PrimFloat(32), "float") => {}
        (&Type::PrimFloat(64), "double") => {}
        (&Type::PrimSigned(16), "__int16") => {}
        (&Type::PrimSigned(16), "short") => {}
        (&Type::PrimSigned(32), "__int32") => {}
        (&Type::PrimSigned(32), "const int") => {}
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
        (&Type::Ptr(&Type::PrimUnsigned(32)), "unsigned int*") => {}
        (&Type::Ptr(&Type::PrimUnsigned(64)), "unsigned __int64*") => {}
        (&Type::Ptr(&Type::PrimUnsigned(8)), "const void*") => {}
        (&Type::Ptr(&Type::PrimUnsigned(8)), "void*") => {}

        (&Type::M64, "__m64")
        | (&Type::Ptr(&Type::M64), "__m64*") => {}

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

        // These two intrinsics return a 16-bit element but in Intel's
        // intrinsics they're listed as returning an `int`.
        (&Type::PrimSigned(16), "int") if intrinsic == "_mm_extract_pi16" => {}
        (&Type::PrimSigned(16), "int") if intrinsic == "_m_pextrw" => {}

        // This intrinsic takes an `i8` to get inserted into an i8 vector, but
        // Intel says the argument is i32...
        (&Type::PrimSigned(8), "int") if intrinsic == "_mm_insert_epi8" => {}

        // This is a macro (?) in C which seems to mutate its arguments, but
        // that means that we're taking pointers to arguments in rust
        // as we're not exposing it as a macro.
        (&Type::Ptr(&Type::M128), "__m128")
            if intrinsic == "_MM_TRANSPOSE4_PS" => {}

        // These intrinsics return an `int` in C but they're always either the
        // bit 1 or 0 so we switch it to returning `bool` in rust
        (&Type::Bool, "int")
            if intrinsic.starts_with("_mm_comi")
                && intrinsic.ends_with("_sd") => {}
        (&Type::Bool, "int")
            if intrinsic.starts_with("_mm_ucomi")
                && intrinsic.ends_with("_sd") => {}

        _ => panic!(
            "failed to equate: `{}` and {:?} for {}",
            intel, t, intrinsic
        ),
    }
}
