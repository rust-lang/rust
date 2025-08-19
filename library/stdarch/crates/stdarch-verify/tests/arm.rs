#![allow(unused)]

use std::collections::HashMap;

use serde::Deserialize;

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
static I16: Type = Type::PrimSigned(16);
static I32: Type = Type::PrimSigned(32);
static I64: Type = Type::PrimSigned(64);
static I8: Type = Type::PrimSigned(8);
static U16: Type = Type::PrimUnsigned(16);
static U32: Type = Type::PrimUnsigned(32);
static U64: Type = Type::PrimUnsigned(64);
static U8: Type = Type::PrimUnsigned(8);
static NEVER: Type = Type::Never;
static GENERICT: Type = Type::GenericParam("T");
static GENERICU: Type = Type::GenericParam("U");

static F16X4: Type = Type::F(16, 4, 1);
static F16X4X2: Type = Type::F(16, 4, 2);
static F16X4X3: Type = Type::F(16, 4, 3);
static F16X4X4: Type = Type::F(16, 4, 4);
static F16X8: Type = Type::F(16, 8, 1);
static F16X8X2: Type = Type::F(16, 8, 2);
static F16X8X3: Type = Type::F(16, 8, 3);
static F16X8X4: Type = Type::F(16, 8, 4);
static F32X2: Type = Type::F(32, 2, 1);
static F32X2X2: Type = Type::F(32, 2, 2);
static F32X2X3: Type = Type::F(32, 2, 3);
static F32X2X4: Type = Type::F(32, 2, 4);
static F32X4: Type = Type::F(32, 4, 1);
static F32X4X2: Type = Type::F(32, 4, 2);
static F32X4X3: Type = Type::F(32, 4, 3);
static F32X4X4: Type = Type::F(32, 4, 4);
static F64X1: Type = Type::F(64, 1, 1);
static F64X1X2: Type = Type::F(64, 1, 2);
static F64X1X3: Type = Type::F(64, 1, 3);
static F64X1X4: Type = Type::F(64, 1, 4);
static F64X2: Type = Type::F(64, 2, 1);
static F64X2X2: Type = Type::F(64, 2, 2);
static F64X2X3: Type = Type::F(64, 2, 3);
static F64X2X4: Type = Type::F(64, 2, 4);
static I16X2: Type = Type::I(16, 2, 1);
static I16X4: Type = Type::I(16, 4, 1);
static I16X4X2: Type = Type::I(16, 4, 2);
static I16X4X3: Type = Type::I(16, 4, 3);
static I16X4X4: Type = Type::I(16, 4, 4);
static I16X8: Type = Type::I(16, 8, 1);
static I16X8X2: Type = Type::I(16, 8, 2);
static I16X8X3: Type = Type::I(16, 8, 3);
static I16X8X4: Type = Type::I(16, 8, 4);
static I32X2: Type = Type::I(32, 2, 1);
static I32X2X2: Type = Type::I(32, 2, 2);
static I32X2X3: Type = Type::I(32, 2, 3);
static I32X2X4: Type = Type::I(32, 2, 4);
static I32X4: Type = Type::I(32, 4, 1);
static I32X4X2: Type = Type::I(32, 4, 2);
static I32X4X3: Type = Type::I(32, 4, 3);
static I32X4X4: Type = Type::I(32, 4, 4);
static I64X1: Type = Type::I(64, 1, 1);
static I64X1X2: Type = Type::I(64, 1, 2);
static I64X1X3: Type = Type::I(64, 1, 3);
static I64X1X4: Type = Type::I(64, 1, 4);
static I64X2: Type = Type::I(64, 2, 1);
static I64X2X2: Type = Type::I(64, 2, 2);
static I64X2X3: Type = Type::I(64, 2, 3);
static I64X2X4: Type = Type::I(64, 2, 4);
static I8X16: Type = Type::I(8, 16, 1);
static I8X16X2: Type = Type::I(8, 16, 2);
static I8X16X3: Type = Type::I(8, 16, 3);
static I8X16X4: Type = Type::I(8, 16, 4);
static I8X4: Type = Type::I(8, 4, 1);
static I8X8: Type = Type::I(8, 8, 1);
static I8X8X2: Type = Type::I(8, 8, 2);
static I8X8X3: Type = Type::I(8, 8, 3);
static I8X8X4: Type = Type::I(8, 8, 4);
static P128: Type = Type::PrimPoly(128);
static P16: Type = Type::PrimPoly(16);
static P16X4X2: Type = Type::P(16, 4, 2);
static P16X4X3: Type = Type::P(16, 4, 3);
static P16X4X4: Type = Type::P(16, 4, 4);
static P16X8X2: Type = Type::P(16, 8, 2);
static P16X8X3: Type = Type::P(16, 8, 3);
static P16X8X4: Type = Type::P(16, 8, 4);
static P64: Type = Type::PrimPoly(64);
static P64X1X2: Type = Type::P(64, 1, 2);
static P64X1X3: Type = Type::P(64, 1, 3);
static P64X1X4: Type = Type::P(64, 1, 4);
static P64X2X2: Type = Type::P(64, 2, 2);
static P64X2X3: Type = Type::P(64, 2, 3);
static P64X2X4: Type = Type::P(64, 2, 4);
static P8: Type = Type::PrimPoly(8);
static POLY16X4: Type = Type::P(16, 4, 1);
static POLY16X8: Type = Type::P(16, 8, 1);
static POLY64X1: Type = Type::P(64, 1, 1);
static POLY64X2: Type = Type::P(64, 2, 1);
static POLY8X16: Type = Type::P(8, 16, 1);
static POLY8X16X2: Type = Type::P(8, 16, 2);
static POLY8X16X3: Type = Type::P(8, 16, 3);
static POLY8X16X4: Type = Type::P(8, 16, 4);
static POLY8X8: Type = Type::P(8, 8, 1);
static POLY8X8X2: Type = Type::P(8, 8, 2);
static POLY8X8X3: Type = Type::P(8, 8, 3);
static POLY8X8X4: Type = Type::P(8, 8, 4);
static U16X4: Type = Type::U(16, 4, 1);
static U16X4X2: Type = Type::U(16, 4, 2);
static U16X4X3: Type = Type::U(16, 4, 3);
static U16X4X4: Type = Type::U(16, 4, 4);
static U16X8: Type = Type::U(16, 8, 1);
static U16X8X2: Type = Type::U(16, 8, 2);
static U16X8X3: Type = Type::U(16, 8, 3);
static U16X8X4: Type = Type::U(16, 8, 4);
static U32X2: Type = Type::U(32, 2, 1);
static U32X2X2: Type = Type::U(32, 2, 2);
static U32X2X3: Type = Type::U(32, 2, 3);
static U32X2X4: Type = Type::U(32, 2, 4);
static U32X4: Type = Type::U(32, 4, 1);
static U32X4X2: Type = Type::U(32, 4, 2);
static U32X4X3: Type = Type::U(32, 4, 3);
static U32X4X4: Type = Type::U(32, 4, 4);
static U64X1: Type = Type::U(64, 1, 1);
static U64X1X2: Type = Type::U(64, 1, 2);
static U64X1X3: Type = Type::U(64, 1, 3);
static U64X1X4: Type = Type::U(64, 1, 4);
static U64X2: Type = Type::U(64, 2, 1);
static U64X2X2: Type = Type::U(64, 2, 2);
static U64X2X3: Type = Type::U(64, 2, 3);
static U64X2X4: Type = Type::U(64, 2, 4);
static U8X16: Type = Type::U(8, 16, 1);
static U8X16X2: Type = Type::U(8, 16, 2);
static U8X16X3: Type = Type::U(8, 16, 3);
static U8X16X4: Type = Type::U(8, 16, 4);
static U8X8: Type = Type::U(8, 8, 1);
static U8X4: Type = Type::U(8, 4, 1);
static U8X8X2: Type = Type::U(8, 8, 2);
static U8X8X3: Type = Type::U(8, 8, 3);
static U8X8X4: Type = Type::U(8, 8, 4);

#[derive(Debug, Copy, Clone, PartialEq)]
enum Type {
    PrimFloat(u8),
    PrimSigned(u8),
    PrimUnsigned(u8),
    PrimPoly(u8),
    MutPtr(&'static Type),
    ConstPtr(&'static Type),
    GenericParam(&'static str),
    I(u8, u8, u8),
    U(u8, u8, u8),
    P(u8, u8, u8),
    F(u8, u8, u8),
    Never,
}

stdarch_verify::arm_functions!(static FUNCTIONS);

macro_rules! bail {
    ($($t:tt)*) => (return Err(format!($($t)*)))
}

#[test]
fn verify_all_signatures() {
    // Reference: https://developer.arm.com/architectures/instruction-sets/intrinsics
    let json = include_bytes!("../../../intrinsics_data/arm_intrinsics.json");
    let intrinsics: Vec<JsonIntrinsic> = serde_json::from_slice(json).unwrap();
    let map = parse_intrinsics(intrinsics);

    let mut all_valid = true;
    for rust in FUNCTIONS {
        if !rust.has_test {
            let skip = [
                "vaddq_s64",
                "vaddq_u64",
                "vrsqrte_f32",
                "vtbl1_s8",
                "vtbl1_u8",
                "vtbl1_p8",
                "vtbl2_s8",
                "vtbl2_u8",
                "vtbl2_p8",
                "vtbl3_s8",
                "vtbl3_u8",
                "vtbl3_p8",
                "vtbl4_s8",
                "vtbl4_u8",
                "vtbl4_p8",
                "vtbx1_s8",
                "vtbx1_u8",
                "vtbx1_p8",
                "vtbx2_s8",
                "vtbx2_u8",
                "vtbx2_p8",
                "vtbx3_s8",
                "vtbx3_u8",
                "vtbx3_p8",
                "vtbx4_s8",
                "vtbx4_u8",
                "vtbx4_p8",
                "udf",
                "_clz_u8",
                "_clz_u16",
                "_clz_u32",
                "_rbit_u32",
                "_rev_u16",
                "_rev_u32",
                "__breakpoint",
                "vpminq_f32",
                "vpminq_f64",
                "vpmaxq_f32",
                "vpmaxq_f64",
                "vcombine_s8",
                "vcombine_s16",
                "vcombine_s32",
                "vcombine_s64",
                "vcombine_u8",
                "vcombine_u16",
                "vcombine_u32",
                "vcombine_u64",
                "vcombine_p64",
                "vcombine_f32",
                "vcombine_p8",
                "vcombine_p16",
                "vcombine_f64",
                "vtbl1_s8",
                "vtbl1_u8",
                "vtbl1_p8",
                "vtbl2_s8",
                "vtbl2_u8",
                "vtbl2_p8",
                "vtbl3_s8",
                "vtbl3_u8",
                "vtbl3_p8",
                "vtbl4_s8",
                "vtbl4_u8",
                "vtbl4_p8",
                "vtbx1_s8",
                "vtbx1_u8",
                "vtbx1_p8",
                "vtbx2_s8",
                "vtbx2_u8",
                "vtbx2_p8",
                "vtbx3_s8",
                "vtbx3_u8",
                "vtbx3_p8",
                "vtbx4_s8",
                "vtbx4_u8",
                "vtbx4_p8",
                "vqtbl1_s8",
                "vqtbl1q_s8",
                "vqtbl1_u8",
                "vqtbl1q_u8",
                "vqtbl1_p8",
                "vqtbl1q_p8",
                "vqtbx1_s8",
                "vqtbx1q_s8",
                "vqtbx1_u8",
                "vqtbx1q_u8",
                "vqtbx1_p8",
                "vqtbx1q_p8",
                "vqtbl2_s8",
                "vqtbl2q_s8",
                "vqtbl2_u8",
                "vqtbl2q_u8",
                "vqtbl2_p8",
                "vqtbl2q_p8",
                "vqtbx2_s8",
                "vqtbx2q_s8",
                "vqtbx2_u8",
                "vqtbx2q_u8",
                "vqtbx2_p8",
                "vqtbx2q_p8",
                "vqtbl3_s8",
                "vqtbl3q_s8",
                "vqtbl3_u8",
                "vqtbl3q_u8",
                "vqtbl3_p8",
                "vqtbl3q_p8",
                "vqtbx3_s8",
                "vqtbx3q_s8",
                "vqtbx3_u8",
                "vqtbx3q_u8",
                "vqtbx3_p8",
                "vqtbx3q_p8",
                "vqtbl4_s8",
                "vqtbl4q_s8",
                "vqtbl4_u8",
                "vqtbl4q_u8",
                "vqtbl4_p8",
                "vqtbl4q_p8",
                "vqtbx4_s8",
                "vqtbx4q_s8",
                "vqtbx4_u8",
                "vqtbx4q_u8",
                "vqtbx4_p8",
                "vqtbx4q_p8",
                "brk",
                "_rev_u64",
                "_clz_u64",
                "_rbit_u64",
                "_cls_u32",
                "_cls_u64",
                "_prefetch",
                "vsli_n_s8",
                "vsliq_n_s8",
                "vsli_n_s16",
                "vsliq_n_s16",
                "vsli_n_s32",
                "vsliq_n_s32",
                "vsli_n_s64",
                "vsliq_n_s64",
                "vsli_n_u8",
                "vsliq_n_u8",
                "vsli_n_u16",
                "vsliq_n_u16",
                "vsli_n_u32",
                "vsliq_n_u32",
                "vsli_n_u64",
                "vsliq_n_u64",
                "vsli_n_p8",
                "vsliq_n_p8",
                "vsli_n_p16",
                "vsliq_n_p16",
                "vsli_n_p64",
                "vsliq_n_p64",
                "vsri_n_s8",
                "vsriq_n_s8",
                "vsri_n_s16",
                "vsriq_n_s16",
                "vsri_n_s32",
                "vsriq_n_s32",
                "vsri_n_s64",
                "vsriq_n_s64",
                "vsri_n_u8",
                "vsriq_n_u8",
                "vsri_n_u16",
                "vsriq_n_u16",
                "vsri_n_u32",
                "vsriq_n_u32",
                "vsri_n_u64",
                "vsriq_n_u64",
                "vsri_n_p8",
                "vsriq_n_p8",
                "vsri_n_p16",
                "vsriq_n_p16",
                "vsri_n_p64",
                "vsriq_n_p64",
                "__smulbb",
                "__smultb",
                "__smulbt",
                "__smultt",
                "__smulwb",
                "__smulwt",
                "__qadd",
                "__qsub",
                "__qdbl",
                "__smlabb",
                "__smlabt",
                "__smlatb",
                "__smlatt",
                "__smlawb",
                "__smlawt",
                "__qadd8",
                "__qsub8",
                "__qsub16",
                "__qadd16",
                "__qasx",
                "__qsax",
                "__sadd16",
                "__sadd8",
                "__smlad",
                "__smlsd",
                "__sasx",
                "__sel",
                "__shadd8",
                "__shadd16",
                "__shsub8",
                "__usub8",
                "__ssub8",
                "__shsub16",
                "__smuad",
                "__smuadx",
                "__smusd",
                "__smusdx",
                "__usad8",
                "__usada8",
                "__ldrex",
                "__strex",
                "__ldrexb",
                "__strexb",
                "__ldrexh",
                "__strexh",
                "__clrex",
                "__dbg",
            ];
        }

        // Skip some intrinsics that aren't NEON and are located in different
        // places than the whitelists below.
        match rust.name {
            "brk" | "__breakpoint" | "udf" | "_prefetch" => continue,
            _ => {}
        }
        // Skip some intrinsics that are present in GCC and Clang but
        // are missing from the official documentation.
        let skip_intrinsic_verify = [
            "vmov_n_p64",
            "vmovq_n_p64",
            "vreinterpret_p64_s64",
            "vreinterpret_f32_p64",
            "vreinterpretq_f32_p64",
            "vreinterpretq_p64_p128",
            "vreinterpretq_p128_p64",
            "vreinterpretq_f32_p128",
            "vtst_p16",
            "vtstq_p16",
            "__dbg",
        ];
        let arm = match map.get(rust.name) {
            Some(i) => i,
            None => {
                // Skip all these intrinsics as they're not listed in NEON
                // descriptions online.
                //
                // TODO: we still need to verify these intrinsics or find a
                // reference for them, need to figure out where though!
                if !rust.file.ends_with("dsp.rs\"")
                    && !rust.file.ends_with("sat.rs\"")
                    && !rust.file.ends_with("simd32.rs\"")
                    && !rust.file.ends_with("v6.rs\"")
                    && !rust.file.ends_with("v7.rs\"")
                    && !rust.file.ends_with("v8.rs\"")
                    && !rust.file.ends_with("tme.rs\"")
                    && !rust.file.ends_with("mte.rs\"")
                    && !rust.file.ends_with("ex.rs\"")
                    && !skip_intrinsic_verify.contains(&rust.name)
                {
                    println!(
                        "missing arm definition for {:?} in {}",
                        rust.name, rust.file
                    );
                    all_valid = false;
                }
                continue;
            }
        };

        if let Err(e) = matches(rust, arm) {
            println!("failed to verify `{}`", rust.name);
            println!("  * {e}");
            all_valid = false;
        }
    }
    assert!(all_valid);
}

fn matches(rust: &Function, arm: &Intrinsic) -> Result<(), String> {
    if rust.ret != arm.ret.as_ref() {
        bail!("mismatched return value")
    }
    if rust.arguments.len() != arm.arguments.len() {
        bail!("mismatched argument lengths");
    }

    let mut nconst = 0;
    let iter = rust.arguments.iter().zip(&arm.arguments).enumerate();
    for (i, (rust_ty, (arm, arm_const))) in iter {
        if *rust_ty != arm {
            bail!("mismatched arguments: {rust_ty:?} != {arm:?}")
        }
        if *arm_const {
            nconst += 1;
            if !rust.required_const.contains(&i) {
                bail!("argument const mismatch");
            }
        }
    }
    if nconst != rust.required_const.len() {
        bail!("wrong number of const arguments");
    }

    if rust.instrs.is_empty() {
        bail!(
            "instruction not listed for `{}`, but arm lists {:?}",
            rust.name,
            arm.instruction
        );
    } else if false
    // TODO: This instruction checking logic needs work to handle multiple instructions and to only
    // look at aarch64 insructions.
    // The ACLE's listed instructions are a guideline only and compilers have the freedom to use
    // different instructions in dfferent cases which makes this an unreliable testing method. It
    // is of questionable value given the intrinsic test tool.
    {
        for instr in rust.instrs {
            if arm.instruction.starts_with(instr) {
                continue;
            }
            // sometimes arm says `foo` and disassemblers say `vfoo`, or
            // sometimes disassemblers say `vfoo` and arm says `sfoo` or `ffoo`
            if instr.starts_with('v')
                && (arm.instruction.starts_with(&instr[1..])
                    || arm.instruction[1..].starts_with(&instr[1..]))
            {
                continue;
            }
            bail!(
                "arm failed to list `{}` as an instruction for `{}` in {:?}",
                instr,
                rust.name,
                arm.instruction,
            );
        }
    }

    // TODO: verify `target_feature`.

    Ok(())
}

#[derive(PartialEq)]
struct Intrinsic {
    name: String,
    ret: Option<Type>,
    arguments: Vec<(Type, bool)>,
    instruction: String,
}

// These structures are similar to those in json_parser.rs in intrinsics-test
#[derive(Deserialize, Debug)]
struct JsonIntrinsic {
    name: String,
    arguments: Vec<String>,
    return_type: ReturnType,
    #[serde(default)]
    instructions: Vec<Vec<String>>,
}

#[derive(Deserialize, Debug)]
struct ReturnType {
    value: String,
}

fn parse_intrinsics(intrinsics: Vec<JsonIntrinsic>) -> HashMap<String, Intrinsic> {
    let mut ret = HashMap::new();
    for intr in intrinsics.into_iter() {
        let f = parse_intrinsic(intr);
        ret.insert(f.name.clone(), f);
    }
    ret
}

fn parse_intrinsic(mut intr: JsonIntrinsic) -> Intrinsic {
    let name = intr.name;
    let ret = if intr.return_type.value == "void" {
        None
    } else {
        Some(parse_ty(&intr.return_type.value))
    };

    // This ignores multiple instructions and different optional sequences for now to mimic
    // the old HTML scraping behaviour
    let instruction = intr.instructions.swap_remove(0).swap_remove(0);

    let arguments = intr
        .arguments
        .iter()
        .map(|s| {
            let (ty, konst) = match s.strip_prefix("const") {
                Some(stripped) => (stripped.trim_start(), true),
                None => (s.as_str(), false),
            };
            let ty = ty.rsplit_once(' ').unwrap().0;
            (parse_ty(ty), konst)
        })
        .collect::<Vec<_>>();

    Intrinsic {
        name,
        ret,
        instruction,
        arguments,
    }
}

fn parse_ty(s: &str) -> Type {
    let suffix = " const *";
    if let Some(base) = s.strip_suffix(suffix) {
        Type::ConstPtr(parse_ty_base(base))
    } else if let Some(base) = s.strip_suffix(" *") {
        Type::MutPtr(parse_ty_base(base))
    } else {
        *parse_ty_base(s)
    }
}

fn parse_ty_base(s: &str) -> &'static Type {
    match s {
        "float16_t" => &F16,
        "float16x4_t" => &F16X4,
        "float16x4x2_t" => &F16X4X2,
        "float16x4x3_t" => &F16X4X3,
        "float16x4x4_t" => &F16X4X4,
        "float16x8_t" => &F16X8,
        "float16x8x2_t" => &F16X8X2,
        "float16x8x3_t" => &F16X8X3,
        "float16x8x4_t" => &F16X8X4,
        "float32_t" => &F32,
        "float32x2_t" => &F32X2,
        "float32x2x2_t" => &F32X2X2,
        "float32x2x3_t" => &F32X2X3,
        "float32x2x4_t" => &F32X2X4,
        "float32x4_t" => &F32X4,
        "float32x4x2_t" => &F32X4X2,
        "float32x4x3_t" => &F32X4X3,
        "float32x4x4_t" => &F32X4X4,
        "float64_t" => &F64,
        "float64x1_t" => &F64X1,
        "float64x1x2_t" => &F64X1X2,
        "float64x1x3_t" => &F64X1X3,
        "float64x1x4_t" => &F64X1X4,
        "float64x2_t" => &F64X2,
        "float64x2x2_t" => &F64X2X2,
        "float64x2x3_t" => &F64X2X3,
        "float64x2x4_t" => &F64X2X4,
        "int16_t" => &I16,
        "int16x2_t" => &I16X2,
        "int16x4_t" => &I16X4,
        "int16x4x2_t" => &I16X4X2,
        "int16x4x3_t" => &I16X4X3,
        "int16x4x4_t" => &I16X4X4,
        "int16x8_t" => &I16X8,
        "int16x8x2_t" => &I16X8X2,
        "int16x8x3_t" => &I16X8X3,
        "int16x8x4_t" => &I16X8X4,
        "int32_t" | "int" => &I32,
        "int32x2_t" => &I32X2,
        "int32x2x2_t" => &I32X2X2,
        "int32x2x3_t" => &I32X2X3,
        "int32x2x4_t" => &I32X2X4,
        "int32x4_t" => &I32X4,
        "int32x4x2_t" => &I32X4X2,
        "int32x4x3_t" => &I32X4X3,
        "int32x4x4_t" => &I32X4X4,
        "int64_t" => &I64,
        "int64x1_t" => &I64X1,
        "int64x1x2_t" => &I64X1X2,
        "int64x1x3_t" => &I64X1X3,
        "int64x1x4_t" => &I64X1X4,
        "int64x2_t" => &I64X2,
        "int64x2x2_t" => &I64X2X2,
        "int64x2x3_t" => &I64X2X3,
        "int64x2x4_t" => &I64X2X4,
        "int8_t" => &I8,
        "int8x16_t" => &I8X16,
        "int8x16x2_t" => &I8X16X2,
        "int8x16x3_t" => &I8X16X3,
        "int8x16x4_t" => &I8X16X4,
        "int8x4_t" => &I8X4,
        "int8x8_t" => &I8X8,
        "int8x8x2_t" => &I8X8X2,
        "int8x8x3_t" => &I8X8X3,
        "int8x8x4_t" => &I8X8X4,
        "poly128_t" => &P128,
        "poly16_t" => &P16,
        "poly16x4_t" => &POLY16X4,
        "poly16x4x2_t" => &P16X4X2,
        "poly16x4x3_t" => &P16X4X3,
        "poly16x4x4_t" => &P16X4X4,
        "poly16x8_t" => &POLY16X8,
        "poly16x8x2_t" => &P16X8X2,
        "poly16x8x3_t" => &P16X8X3,
        "poly16x8x4_t" => &P16X8X4,
        "poly64_t" => &P64,
        "poly64x1_t" => &POLY64X1,
        "poly64x1x2_t" => &P64X1X2,
        "poly64x1x3_t" => &P64X1X3,
        "poly64x1x4_t" => &P64X1X4,
        "poly64x2_t" => &POLY64X2,
        "poly64x2x2_t" => &P64X2X2,
        "poly64x2x3_t" => &P64X2X3,
        "poly64x2x4_t" => &P64X2X4,
        "poly8_t" => &P8,
        "poly8x16_t" => &POLY8X16,
        "poly8x16x2_t" => &POLY8X16X2,
        "poly8x16x3_t" => &POLY8X16X3,
        "poly8x16x4_t" => &POLY8X16X4,
        "poly8x8_t" => &POLY8X8,
        "poly8x8x2_t" => &POLY8X8X2,
        "poly8x8x3_t" => &POLY8X8X3,
        "poly8x8x4_t" => &POLY8X8X4,
        "uint16_t" => &U16,
        "uint16x4_t" => &U16X4,
        "uint16x4x2_t" => &U16X4X2,
        "uint16x4x3_t" => &U16X4X3,
        "uint16x4x4_t" => &U16X4X4,
        "uint16x8_t" => &U16X8,
        "uint16x8x2_t" => &U16X8X2,
        "uint16x8x3_t" => &U16X8X3,
        "uint16x8x4_t" => &U16X8X4,
        "uint32_t" => &U32,
        "uint32x2_t" => &U32X2,
        "uint32x2x2_t" => &U32X2X2,
        "uint32x2x3_t" => &U32X2X3,
        "uint32x2x4_t" => &U32X2X4,
        "uint32x4_t" => &U32X4,
        "uint32x4x2_t" => &U32X4X2,
        "uint32x4x3_t" => &U32X4X3,
        "uint32x4x4_t" => &U32X4X4,
        "uint64_t" => &U64,
        "uint64x1_t" => &U64X1,
        "uint64x1x2_t" => &U64X1X2,
        "uint64x1x3_t" => &U64X1X3,
        "uint64x1x4_t" => &U64X1X4,
        "uint64x2_t" => &U64X2,
        "uint64x2x2_t" => &U64X2X2,
        "uint64x2x3_t" => &U64X2X3,
        "uint64x2x4_t" => &U64X2X4,
        "uint8_t" => &U8,
        "uint8x16_t" => &U8X16,
        "uint8x16x2_t" => &U8X16X2,
        "uint8x16x3_t" => &U8X16X3,
        "uint8x16x4_t" => &U8X16X4,
        "uint8x8_t" => &U8X8,
        "uint8x8x2_t" => &U8X8X2,
        "uint8x8x3_t" => &U8X8X3,
        "uint8x8x4_t" => &U8X8X4,

        _ => panic!("failed to parse json type {s:?}"),
    }
}
