mod intrinsic;
mod json_parser;
mod types;

use crate::common::SupportedArchitecture;
use crate::common::cli::{CcArgStyle, ProcessedCli};
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::{SimdLen, TypeKind};
use intrinsic::ArmType;
use json_parser::get_neon_intrinsics;

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
"#;
    const RUST_PRELUDE: &str = RUST_PRELUDE;

    fn c_compiler_flags(&self, cli_options: &ProcessedCli) -> Vec<&str> {
        // GCC uses an extra `-` in the arch name
        match cli_options.cc_arg_style {
            CcArgStyle::Clang => vec!["-march=armv8.6a+crypto+crc+dotprod+fp16"],
            // SVE tests aren't run under GCC so there are no target features added for SVE
            CcArgStyle::Gcc => vec!["-march=armv8.6-a+crypto+crc+dotprod+fp16+sha3+sm4"],
        }
    }

    fn create(cli_options: &ProcessedCli) -> Self {
        let big_endian = cli_options.target.starts_with("aarch64_be");
        let a32 = cli_options.target.starts_with("armv7");
        let mut intrinsics =
            get_neon_intrinsics(&cli_options.filename).expect("Error parsing input file");

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
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(faminmax))]
#![feature(stdarch_neon_f16)]

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
use core_arch::arch::aarch64::*;

#[cfg(target_arch = "arm")]
use core_arch::arch::arm::*;
"#;
