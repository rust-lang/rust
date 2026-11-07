//! Verification of MIPS MSA intrinsics
#![allow(unused, non_upper_case_globals, clippy::single_match)]

// This file is obtained from
// https://gcc.gnu.org/onlinedocs//gcc/MIPS-SIMD-Architecture-Built-in-Functions.html
static HEADER: &str = include_str!("../mips-msa.h");

stdarch_verify::mips_functions!(static FUNCTIONS);

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
static NEVER: Type = Type::Never;
static TUPLE: Type = Type::Tuple;
static v16i8: Type = Type::I(8, 16, 1);
static v8i16: Type = Type::I(16, 8, 1);
static v4i32: Type = Type::I(32, 4, 1);
static v2i64: Type = Type::I(64, 2, 1);
static v16u8: Type = Type::U(8, 16, 1);
static v8u16: Type = Type::U(16, 8, 1);
static v4u32: Type = Type::U(32, 4, 1);
static v2u64: Type = Type::U(64, 2, 1);
static v8f16: Type = Type::F(16, 8, 1);
static v4f32: Type = Type::F(32, 4, 1);
static v2f64: Type = Type::F(64, 2, 1);

#[derive(Debug, Copy, Clone, PartialEq)]
enum Type {
    PrimFloat(u8),
    PrimSigned(u8),
    PrimUnsigned(u8),
    PrimPoly(u8),
    MutPtr(&'static Type),
    ConstPtr(&'static Type),
    Tuple,
    I(u8, u8, u8),
    U(u8, u8, u8),
    P(u8, u8, u8),
    F(u8, u8, u8),
    Never,
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[allow(non_camel_case_types)]
enum MsaTy {
    v16i8,
    v8i16,
    v4i32,
    v2i64,
    v16u8,
    v8u16,
    v4u32,
    v2u64,
    v8f16,
    v4f32,
    v2f64,
    imm0_1,
    imm0_3,
    imm0_7,
    imm0_15,
    imm0_31,
    imm0_63,
    imm0_255,
    imm_n16_15,
    imm_n512_511,
    imm_n1024_1022,
    imm_n2048_2044,
    imm_n4096_4088,
    i32,
    u32,
    i64,
    u64,
    Void,
    MutVoidPtr,
}

impl<'a> From<&'a str> for MsaTy {
    fn from(s: &'a str) -> MsaTy {
        match s {
            "v16i8" => MsaTy::v16i8,
            "v8i16" => MsaTy::v8i16,
            "v4i32" => MsaTy::v4i32,
            "v2i64" => MsaTy::v2i64,
            "v16u8" => MsaTy::v16u8,
            "v8u16" => MsaTy::v8u16,
            "v4u32" => MsaTy::v4u32,
            "v2u64" => MsaTy::v2u64,
            "v8f16" => MsaTy::v8f16,
            "v4f32" => MsaTy::v4f32,
            "v2f64" => MsaTy::v2f64,
            "imm0_1" => MsaTy::imm0_1,
            "imm0_3" => MsaTy::imm0_3,
            "imm0_7" => MsaTy::imm0_7,
            "imm0_15" => MsaTy::imm0_15,
            "imm0_31" => MsaTy::imm0_31,
            "imm0_63" => MsaTy::imm0_63,
            "imm0_255" => MsaTy::imm0_255,
            "imm_n16_15" => MsaTy::imm_n16_15,
            "imm_n512_511" => MsaTy::imm_n512_511,
            "imm_n1024_1022" => MsaTy::imm_n1024_1022,
            "imm_n2048_2044" => MsaTy::imm_n2048_2044,
            "imm_n4096_4088" => MsaTy::imm_n4096_4088,
            "i32" => MsaTy::i32,
            "u32" => MsaTy::u32,
            "i64" => MsaTy::i64,
            "u64" => MsaTy::u64,
            "void" => MsaTy::Void,
            "void *" => MsaTy::MutVoidPtr,
            v => panic!("unknown ty: \"{v}\""),
        }
    }
}

#[derive(Debug, Clone)]
struct MsaIntrinsic {
    id: String,
    arg_tys: Vec<MsaTy>,
    ret_ty: MsaTy,
    instruction: String,
}

struct NoneError;

impl std::convert::TryFrom<&'static str> for MsaIntrinsic {
    // The intrinsics are just C function declarations of the form:
    // $ret_ty __builtin_${fn_id}($($arg_ty),*);
    type Error = NoneError;
    fn try_from(line: &'static str) -> Result<Self, Self::Error> {
        return inner(line).ok_or(NoneError);

        fn inner(line: &'static str) -> Option<MsaIntrinsic> {
            let first_whitespace = line.find(char::is_whitespace)?;
            let ret_ty = &line[0..first_whitespace];
            let ret_ty = MsaTy::from(ret_ty);

            let first_parentheses = line.find('(')?;
            assert!(first_parentheses > first_whitespace);
            let id = &line[first_whitespace + 1..first_parentheses].trim();
            assert!(id.starts_with("__builtin"));
            let mut id_str = "_".to_string();
            id_str += &id[9..];
            let id = id_str;

            let mut arg_tys = Vec::new();

            let last_parentheses = line.find(')')?;
            for arg in line[first_parentheses + 1..last_parentheses].split(',') {
                let arg = arg.trim();
                arg_tys.push(MsaTy::from(arg));
            }

            // The instruction is the intrinsic name without the __msa_ prefix.
            let instruction = &id[6..];
            let mut instruction = instruction.to_string();
            // With all underscores but the first one replaced with a `.`
            if let Some(first_underscore) = instruction.find('_') {
                let postfix = instruction[first_underscore + 1..].replace('_', ".");
                instruction = instruction[0..=first_underscore].to_string();
                instruction += &postfix;
            }

            Some(MsaIntrinsic {
                id,
                ret_ty,
                arg_tys,
                instruction,
            })
        }
    }
}

#[test]
fn verify_all_signatures() {
    // Parse the C intrinsic header file:
    let mut intrinsics = std::collections::HashMap::<String, MsaIntrinsic>::new();
    for line in HEADER.lines() {
        if line.is_empty() {
            continue;
        }

        use std::convert::TryFrom;
        let intrinsic: MsaIntrinsic =
            TryFrom::try_from(line).unwrap_or_else(|_| panic!("failed to parse line: \"{line}\""));
        assert!(!intrinsics.contains_key(&intrinsic.id));
        intrinsics.insert(intrinsic.id.clone(), intrinsic);
    }

    let mut all_valid = true;
    for rust in FUNCTIONS {
        if !rust.has_test {
            let skip = [
                "__msa_ceqi_d",
                "__msa_cfcmsa",
                "__msa_clei_s_d",
                "__msa_clti_s_d",
                "__msa_ctcmsa",
                "__msa_ldi_d",
                "__msa_maxi_s_d",
                "__msa_mini_s_d",
                "break_",
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

        // Skip some intrinsics that aren't part of MSA
        match rust.name {
            "break_" => continue,
            _ => {}
        }
        let mips = match intrinsics.get(rust.name) {
            Some(i) => i,
            None => {
                eprintln!(
                    "missing mips definition for {:?} in {}",
                    rust.name, rust.file
                );
                all_valid = false;
                continue;
            }
        };

        if let Err(e) = matches(rust, mips) {
            println!("failed to verify `{}`", rust.name);
            println!("  * {e}");
            all_valid = false;
        }
    }
    assert!(all_valid);
}

fn matches(rust: &Function, mips: &MsaIntrinsic) -> Result<(), String> {
    macro_rules! bail {
        ($($t:tt)*) => (return Err(format!($($t)*)))
    }

    if rust.ret.is_none() && mips.ret_ty != MsaTy::Void {
        bail!("mismatched return value")
    }

    if rust.arguments.len() != mips.arg_tys.len() {
        bail!("mismatched argument lengths");
    }

    let mut nconst = 0;
    for (i, (rust_arg, mips_arg)) in rust.arguments.iter().zip(mips.arg_tys.iter()).enumerate() {
        match mips_arg {
            MsaTy::v16i8 if **rust_arg == v16i8 => (),
            MsaTy::v8i16 if **rust_arg == v8i16 => (),
            MsaTy::v4i32 if **rust_arg == v4i32 => (),
            MsaTy::v2i64 if **rust_arg == v2i64 => (),
            MsaTy::v16u8 if **rust_arg == v16u8 => (),
            MsaTy::v8u16 if **rust_arg == v8u16 => (),
            MsaTy::v4u32 if **rust_arg == v4u32 => (),
            MsaTy::v2u64 if **rust_arg == v2u64 => (),
            MsaTy::v4f32 if **rust_arg == v4f32 => (),
            MsaTy::v2f64 if **rust_arg == v2f64 => (),
            MsaTy::imm0_1
            | MsaTy::imm0_3
            | MsaTy::imm0_7
            | MsaTy::imm0_15
            | MsaTy::imm0_31
            | MsaTy::imm0_63
            | MsaTy::imm0_255
            | MsaTy::imm_n16_15
            | MsaTy::imm_n512_511
            | MsaTy::imm_n1024_1022
            | MsaTy::imm_n2048_2044
            | MsaTy::imm_n4096_4088
                if **rust_arg == I32 => {}
            MsaTy::i32 if **rust_arg == I32 => (),
            MsaTy::i64 if **rust_arg == I64 => (),
            MsaTy::u32 if **rust_arg == U32 => (),
            MsaTy::u64 if **rust_arg == U64 => (),
            MsaTy::MutVoidPtr if **rust_arg == Type::MutPtr(&U8) => (),
            m => bail!(
                "mismatched argument \"{}\"= \"{:?}\" != \"{:?}\"",
                i,
                m,
                *rust_arg
            ),
        }

        let is_const = matches!(
            mips_arg,
            MsaTy::imm0_1
                | MsaTy::imm0_3
                | MsaTy::imm0_7
                | MsaTy::imm0_15
                | MsaTy::imm0_31
                | MsaTy::imm0_63
                | MsaTy::imm0_255
                | MsaTy::imm_n16_15
                | MsaTy::imm_n512_511
                | MsaTy::imm_n1024_1022
                | MsaTy::imm_n2048_2044
                | MsaTy::imm_n4096_4088
        );
        if is_const {
            nconst += 1;
            if !rust.required_const.contains(&i) {
                bail!("argument const mismatch");
            }
        }
    }

    if nconst != rust.required_const.len() {
        bail!("wrong number of const arguments");
    }

    if rust.target_feature != Some("msa") {
        bail!("wrong target_feature");
    }

    if !rust.instrs.is_empty() {
        // Normalize slightly to get rid of assembler differences
        let actual = rust.instrs[0].replace('.', "_");
        let expected = mips.instruction.replace('.', "_");
        if actual != expected {
            bail!(
                "wrong instruction: \"{}\" != \"{}\"",
                rust.instrs[0],
                mips.instruction
            );
        }
    } else {
        bail!(
            "missing assert_instr for \"{}\" (should be \"{}\")",
            mips.id,
            mips.instruction
        );
    }

    Ok(())
}
