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
    target_feature: &'static str,
    instrs: &'static [&'static str],
}

static BOOL: Type = Type::Bool;
static F32: Type = Type::PrimFloat(32);
static F32x4: Type = Type::Float(32, 4);
static F32x8: Type = Type::Float(32, 8);
static F64: Type = Type::PrimFloat(64);
static F64x2: Type = Type::Float(64, 2);
static F64x4: Type = Type::Float(64, 4);
static I16: Type = Type::PrimSigned(16);
static I16x16: Type = Type::Signed(16, 16);
static I16x4: Type = Type::Signed(16, 4);
static I16x8: Type = Type::Signed(16, 8);
static I32: Type = Type::PrimSigned(32);
static I32x2: Type = Type::Signed(32, 2);
static I32x4: Type = Type::Signed(32, 4);
static I32x8: Type = Type::Signed(32, 8);
static I64: Type = Type::PrimSigned(64);
static I64x2: Type = Type::Signed(64, 2);
static I64x4: Type = Type::Signed(64, 4);
static I8: Type = Type::PrimSigned(8);
static I8x16: Type = Type::Signed(8, 16);
static I8x32: Type = Type::Signed(8, 32);
static I8x8: Type = Type::Signed(8, 8);
static U16: Type = Type::PrimUnsigned(16);
static U16x16: Type = Type::Unsigned(16, 16);
// static U16x4: Type = Type::Unsigned(16, 4);
static U16x8: Type = Type::Unsigned(16, 8);
static U32: Type = Type::PrimUnsigned(32);
static U32x2: Type = Type::Unsigned(32, 2);
static U32x4: Type = Type::Unsigned(32, 4);
static U32x8: Type = Type::Unsigned(32, 8);
static U64: Type = Type::PrimUnsigned(64);
static U64x2: Type = Type::Unsigned(64, 2);
static U64x4: Type = Type::Unsigned(64, 4);
static U8: Type = Type::PrimUnsigned(8);
static U8x16: Type = Type::Unsigned(8, 16);
static U8x32: Type = Type::Unsigned(8, 32);
// static U8x8: Type = Type::Unsigned(8, 8);

#[derive(Debug)]
enum Type {
    Float(u8, u8),
    PrimFloat(u8),
    PrimSigned(u8),
    PrimUnsigned(u8),
    Ptr(&'static Type),
    Signed(u8, u8),
    Unsigned(u8, u8),
    Bool,
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

#[derive(Deserialize)]
struct Instruction {
    name: String,
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
        // This intrinsic has multiple definitions in the XML, so just ignore
        // it.
        if intrinsic.name == "_mm_prefetch" {
            continue;
        }

        // These'll need to get added eventually, but right now they have some
        // duplicate names in the XML which we're not dealing with yet
        if intrinsic.tech == "AVX-512" {
            continue;
        }

        assert!(map.insert(&intrinsic.name[..], intrinsic).is_none());
    }

    for rust in FUNCTIONS {
        // This was ignored above, we ignore it here as well.
        if rust.name == "_mm_prefetch" {
            continue;
        }

        // these are all AMD-specific intrinsics
        if rust.target_feature.contains("sse4a")
            || rust.target_feature.contains("tbm")
        {
            continue;
        }

        let intel = match map.get(rust.name) {
            Some(i) => i,
            None => panic!("missing intel definition for {}", rust.name),
        };

        // Verify that all `#[target_feature]` annotations are correct,
        // ensuring that we've actually enabled the right instruction
        // set for this intrinsic.
        assert!(!intel.cpuid.is_empty(), "missing cpuid for {}", rust.name);
        for cpuid in &intel.cpuid {
            // this is needed by _xsave and probably some related intrinsics,
            // but let's just skip it for now.
            if *cpuid == "XSS" {
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

            assert!(
                rust.target_feature.contains(&cpuid),
                "intel cpuid `{}` not in `{}` for {}",
                cpuid,
                rust.target_feature,
                rust.name
            );
        }

        // TODO: we should test this, but it generates too many failures right
        // now
        if false {
            if rust.instrs.is_empty() {
                assert_eq!(
                    intel.instruction.len(),
                    0,
                    "instruction not listed for {}",
                    rust.name
                );

            // If intel doesn't list any instructions and we do then don't
            // bother trying to look for instructions in intel, we've just got
            // some extra assertions on our end.
            } else if !intel.instruction.is_empty() {
                for instr in rust.instrs {
                    assert!(
                        intel
                            .instruction
                            .iter()
                            .any(|a| a.name.starts_with(instr)),
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
            continue;
        }

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

        (&Type::Signed(a, b), "__m128i")
        | (&Type::Unsigned(a, b), "__m128i")
        | (&Type::Ptr(&Type::Signed(a, b)), "__m128i*")
        | (&Type::Ptr(&Type::Unsigned(a, b)), "__m128i*") if a * b == 128 => {}

        (&Type::Signed(a, b), "__m256i")
        | (&Type::Unsigned(a, b), "__m256i")
        | (&Type::Ptr(&Type::Signed(a, b)), "__m256i*")
        | (&Type::Ptr(&Type::Unsigned(a, b)), "__m256i*")
            if (a as u32) * (b as u32) == 256 => {}

        (&Type::Signed(a, b), "__m64")
        | (&Type::Unsigned(a, b), "__m64")
        | (&Type::Ptr(&Type::Signed(a, b)), "__m64*")
        | (&Type::Ptr(&Type::Unsigned(a, b)), "__m64*") if a * b == 64 => {}

        (&Type::Float(32, 4), "__m128") => {}
        (&Type::Ptr(&Type::Float(32, 4)), "__m128*") => {}

        (&Type::Float(64, 2), "__m128d") => {}
        (&Type::Ptr(&Type::Float(64, 2)), "__m128d*") => {}

        (&Type::Float(32, 8), "__m256") => {}
        (&Type::Ptr(&Type::Float(32, 8)), "__m256*") => {}

        (&Type::Float(64, 4), "__m256d") => {}
        (&Type::Ptr(&Type::Float(64, 4)), "__m256d*") => {}

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
        (&Type::Ptr(&Type::Float(32, 4)), "__m128")
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
