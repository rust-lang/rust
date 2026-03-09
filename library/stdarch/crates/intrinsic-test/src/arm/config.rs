pub const NOTICE: &str = "\
// This is a transient test file, not intended for distribution. Some aspects of the
// test are derived from a JSON specification, published under the same license as the
// `intrinsic-test` crate.\n";

pub const PLATFORM_RUST_DEFINITIONS: &str = "";

pub const PLATFORM_RUST_CFGS: &str = r#"
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_neon_intrinsics))]
#![cfg_attr(target_arch = "arm", feature(stdarch_aarch32_crc32))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_fcma))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_dotprod))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_i8mm))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_sm4))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_ftts))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_feat_lut))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_fp8))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(faminmax))]
#![feature(fmt_helpers_for_derive)]
#![feature(stdarch_neon_f16)]

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
use core_arch::arch::aarch64::*;

#[cfg(target_arch = "arm")]
use core_arch::arch::arm::*;
"#;
