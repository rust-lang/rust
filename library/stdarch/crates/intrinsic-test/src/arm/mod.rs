mod intrinsic;
mod json_parser;
mod types;

use crate::common::argument::Argument;
use crate::common::cli::{CcArgStyle, ProcessedCli};
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::{SimdLen, TypeDefinition, TypeKind};
use crate::common::values::test_values_array_name;
use crate::common::{PASSES, PREDICATE_LOCAL, SupportedArchitecture};
use intrinsic::ArmType;
use json_parser::get_intrinsics;

#[derive(PartialEq)]
pub struct Arm(Vec<Intrinsic<Arm>>);

impl SupportedArchitecture for Arm {
    type Type = ArmType;

    fn intrinsics(&self) -> &[Intrinsic<Self>] {
        &self.0
    }

    const NOTICE: &str = r#"
// This is a transient test file, not intended for distribution. Some aspects of the
// test are derived from a JSON specification, published under the same license as the
// `intrinsic-test` crate.
"#;

    const C_PRELUDE: &str = r#"
#include <arm_acle.h>
#include <arm_fp16.h>
#include <arm_neon.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
"#;
    const RUST_PRELUDE: &str = RUST_PRELUDE;

    fn c_compiler_flags(&self, cli_options: &ProcessedCli) -> Vec<&str> {
        // GCC uses an extra `-` in the arch name
        let big_endian = cli_options.target.starts_with("aarch64_be");
        let a32 = cli_options.target.starts_with("armv7");
        match cli_options.cc_arg_style {
            CcArgStyle::Clang if !a32 && !big_endian => vec![
                "-march=armv8.6a+crypto+crc+dotprod+fp16+sve2-aes+sve2-sm4+sve2-sha3+sve2-bitperm+\
                 f32mm+f64mm+sve2p1",
            ],
            CcArgStyle::Clang => vec!["-march=armv8.6a+crypto+crc+dotprod+fp16"],
            // SVE tests aren't run under GCC so there are no target features added for SVE
            CcArgStyle::Gcc => vec!["-march=armv8.6-a+crypto+crc+dotprod+fp16+sha3+sm4"],
        }
    }

    fn create(cli_options: &ProcessedCli) -> Self {
        let big_endian = cli_options.target.starts_with("aarch64_be");
        let a32 = cli_options.target.starts_with("armv7");
        let mut intrinsics =
            get_intrinsics(&cli_options.filename).expect("Error parsing input file");

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));
        intrinsics.dedup();

        let sample_percentage: usize = cli_options.sample_percentage as usize;
        let sample_size = (intrinsics.len() * sample_percentage) / 100;

        let intrinsics = intrinsics
            .into_iter()
            // Skip intrinsics that don't return a value.
            .filter(|i| i.results.kind() != TypeKind::Void)
            // Skip bfloat intrinsics - not currently supported
            .filter(|i| i.results.kind() != TypeKind::BFloat)
            .filter(|i| !i.arguments.iter().any(|a| a.ty.kind() == TypeKind::BFloat))
            // Skip SVE intrinsics that have `f16` - not yet implemented!
            .filter(|i| {
                let has_f16_arg = i
                    .arguments
                    .iter()
                    .any(|a| a.ty.kind() == TypeKind::Float && a.ty.bit_len == Some(16));
                let has_sve_arg = i
                    .arguments
                    .iter()
                    .any(|a| a.ty.num_lanes() == SimdLen::Scalable);
                !(has_f16_arg && has_sve_arg)
            })
            .filter(|i| {
                let has_f16_ret =
                    i.results.kind() == TypeKind::Float && i.results.bit_len == Some(16);
                let has_sve_ret = i.results.num_lanes() == SimdLen::Scalable;
                !(has_f16_ret && has_sve_ret)
            })
            // Skip `svqcvtn{u,}n*_x2` intrinsics - not yet implemented!
            .filter(|i| !(i.name.starts_with("svqcvtn") && i.name.ends_with("_x2")))
            // Skip `svqrshr{u,}n*_x2` intrinsics - not yet implemented!
            .filter(|i| !(i.name.starts_with("svqrshrn") && i.name.ends_with("_x2")))
            .filter(|i| !(i.name.starts_with("svqrshrun") && i.name.ends_with("_x2")))
            // Skip `svclamp*` intrinsics - not yet implemented!
            .filter(|i| !i.name.starts_with("svclamp"))
            // Skip `svdot{_lane,}_{s,u}32_{s,u}16` intrinsics - not yet implemented!
            .filter(|i| {
                i.name != "svdot_lane_u32_u16"
                    && i.name != "svdot_lane_s32_s16"
                    && i.name != "svdot_u32_u16"
                    && i.name != "svdot_s32_s16"
            })
            // Skip `svrevd*` intrinsics - not yet implemented!
            .filter(|i| !i.name.starts_with("svrevd"))
            // Skip `svpsel_lane_b*` intrinsics - not yet implemented!
            .filter(|i| !i.name.starts_with("svpsel_lane_b"))
            // Skip `svundef*` intrinsics - to avoid undefined behaviour in Rust, these return
            // zeroed vectors in Rust, which are inherently going to be different than the
            // undefined vectors returned by the C intrinsics.
            .filter(|i| !i.name.starts_with("svundef"))
            // Skip `sveorv` intrinsics - the code produced by `intrinsic-test` for these
            // miscompiles and the Rust intrinsic call gets replaced by a constant zero (see
            // llvm/llvm-project#203921).
            .filter(|i| !i.name.starts_with("sveorv"))
            // These load intrinsics expect each element in the scalable vector `bases` argument to
            // be able to be cast to a pointer, which we don't support generating tests for yet.
            .filter(|i| !(i.name.starts_with("svld") && i.name.contains("_gather_")))
            // Skip pointers for now, we would probably need to look at the return
            // type to work out how many elements we need to point to.
            .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
            // Skip intrinsics with 128-bit elements (e.g. `p128`)
            .filter(|i| !i.arguments.iter().any(|a| a.ty.inner_size() == 128))
            // Skip intrinsics from `--skip`
            .filter(|i| !cli_options.skip.contains(&i.name))
            // Skip A64-specific intrinsics on A32
            .filter(|i| !(a32 && i.arch_tags == vec!["A64".to_string()]))
            // Skip SVE intrinsics on big endian
            .filter(|i| !(big_endian && (i.extension == "SVE" || i.extension == "SVE2")))
            // Skip SVE intrinsics when testing against GCC as our wrappers run into ICEs
            // See <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=125818>
            .filter(|i| {
                !(matches!(cli_options.cc_arg_style, CcArgStyle::Gcc)
                    && (i.extension == "SVE" || i.extension == "SVE2"))
            })
            .take(sample_size)
            .collect::<Vec<_>>();

        Self(intrinsics)
    }

    fn predicate_function(size: u32) -> String {
        format!("svptrue_b{size}()")
    }

    fn load_call(arg: &Argument<Self>, idx: usize) -> String {
        let name = arg.generate_name();
        let load = arg.ty.load_function();
        let ptr = format!(
            "{vals_name}.as_ptr().add((i+{idx}) % {PASSES}) as _",
            vals_name = test_values_array_name(&arg.ty)
        );

        match arg.ty.num_lanes() {
            // If the load is of a `svbool_t`, then we load a `svint8_t` and
            SimdLen::Scalable if matches!(arg.ty.kind(), TypeKind::Bool) => {
                format!(
                    r#"
let {name} = {load}({PREDICATE_LOCAL}, {ptr});
let {name} = svcmpne_n_s8({PREDICATE_LOCAL}, {name}, 0);
                "#
                )
            }
            // If this load is of a scalable vector, then prepend an additional argument
            // containing the predicate for the load.
            SimdLen::Scalable => format!("let {name} = {load}({PREDICATE_LOCAL}, {ptr});"),
            SimdLen::Fixed(..) => format!("let {name} = {load}({ptr});"),
        }
    }
}

const RUST_PRELUDE: &str = r#"
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_neon_intrinsics))]
#![cfg_attr(target_arch = "arm", feature(stdarch_aarch32_crc32))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_fcma))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_i8mm))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_sm4))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_ftts))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_feat_lut))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_fp8))]
#![cfg_attr(all(any(target_arch = "aarch64", target_arch = "arm64ec"), target_endian = "little"), feature(stdarch_aarch64_sve))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(faminmax))]
#![feature(stdarch_neon_f16)]

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
use core_arch::arch::aarch64::*;

#[cfg(target_arch = "arm")]
use core_arch::arch::arm::*;

#[cfg(all(any(target_arch = "aarch64", target_arch = "arm64ec"), target_endian = "little"))]
const fn svpattern_from_i32(value: i32) -> svpattern {
    match value {
        0 => svpattern::SV_POW2,
        1 => svpattern::SV_VL1,
        2 => svpattern::SV_VL2,
        3 => svpattern::SV_VL3,
        4 => svpattern::SV_VL4,
        5 => svpattern::SV_VL5,
        6 => svpattern::SV_VL6,
        7 => svpattern::SV_VL7,
        8 => svpattern::SV_VL8,
        9 => svpattern::SV_VL16,
        10 => svpattern::SV_VL32,
        11 => svpattern::SV_VL64,
        12 => svpattern::SV_VL128,
        13 => svpattern::SV_VL256,
        29 => svpattern::SV_MUL4,
        30 => svpattern::SV_MUL3,
        31 => svpattern::SV_ALL,
        _ => unreachable!(),
    }
}

#[cfg(all(any(target_arch = "aarch64", target_arch = "arm64ec"), target_endian = "little"))]
const fn svprfop_from_i32(value: i32) -> svprfop {
    match value {
        0 => svprfop::SV_PLDL1KEEP,
        1 => svprfop::SV_PLDL1STRM,
        2 => svprfop::SV_PLDL2KEEP,
        3 => svprfop::SV_PLDL2STRM,
        4 => svprfop::SV_PLDL3KEEP,
        5 => svprfop::SV_PLDL3STRM,
        8 => svprfop::SV_PSTL1KEEP,
        9 => svprfop::SV_PSTL1STRM,
        10 => svprfop::SV_PSTL2KEEP,
        11 => svprfop::SV_PSTL2STRM,
        12 => svprfop::SV_PSTL3KEEP,
        13 => svprfop::SV_PSTL3STRM,
        _ => unreachable!(),
    }
}

macro_rules! debug_print_integral {
    ($($name:ident => ($ty:ty, $svptrue_fn:ident, $svcnt_fn:ident, $svst_fn:ident)),*) => {
        $(
            #[inline]
            #[target_feature(enable = "sve")]
            #[cfg(all(any(target_arch = "aarch64", target_arch = "arm64ec"), target_endian = "little"))]
            pub fn $name(v: $ty) -> String {
                unsafe {
                    let __pred = $svptrue_fn();
                    let __num_elems = $svcnt_fn() as usize;
                    let mut __buf = std::vec::Vec::with_capacity(__num_elems);
                    $svst_fn(__pred, __buf.as_mut_ptr(), v);
                    __buf.set_len(__num_elems);
                    format!(
                        "[{}]",
                        __buf.iter().map(|el| el.to_string()).collect::<Vec<_>>().join(", ")
                    )
                }
            }
        )*
    }
}

debug_print_integral! {
    debug_print_f32 => (svfloat32_t, svptrue_b32, svcntw, svst1_f32),
    debug_print_f64 => (svfloat64_t, svptrue_b64, svcntd, svst1_f64),
    debug_print_s8 => (svint8_t, svptrue_b8, svcntb, svst1_s8),
    debug_print_s16 => (svint16_t, svptrue_b16, svcnth, svst1_s16),
    debug_print_s32 => (svint32_t, svptrue_b32, svcntw, svst1_s32),
    debug_print_s64 => (svint64_t, svptrue_b64, svcntd, svst1_s64),
    debug_print_u8 => (svuint8_t, svptrue_b8, svcntb, svst1_u8),
    debug_print_u16 => (svuint16_t, svptrue_b16, svcnth, svst1_u16),
    debug_print_u32 => (svuint32_t, svptrue_b32, svcntw, svst1_u32),
    debug_print_u64 => (svuint64_t, svptrue_b64, svcntd, svst1_u64)
}

macro_rules! debug_print_bool {
    ($($name:ident => ($ty:ty, $svst_fn:ident, $svdup_fn:ident)),*) => {
        $(
            #[inline]
            #[target_feature(enable = "sve")]
            #[cfg(all(any(target_arch = "aarch64", target_arch = "arm64ec"), target_endian = "little"))]
            pub fn $name(v: $ty) -> String {
                unsafe {
                    let __num_elems = svcntb() as usize;
                    let mut __buf = std::vec::Vec::with_capacity(__num_elems);
                    $svst_fn(v, __buf.as_mut_ptr(), $svdup_fn(1));
                    __buf.set_len(__num_elems);
                    format!(
                        "[{}]",
                        __buf.iter()
                            .map(|el| *el == 1)
                            .map(|el| el.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
        )*
    }
}

debug_print_bool! {
    debug_print_b8 => (svbool_t, svst1_u8, svdup_n_u8),
    debug_print_b16 => (svbool_t, svst1_u16, svdup_n_u16),
    debug_print_b32 => (svbool_t, svst1_u32, svdup_n_u32),
    debug_print_b64 => (svbool_t, svst1_u64, svdup_n_u64)
}
"#;
