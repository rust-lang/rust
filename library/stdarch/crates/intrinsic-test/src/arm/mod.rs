mod config;
mod intrinsic;
mod json_parser;
mod types;

use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use intrinsic::ArmIntrinsicType;
use json_parser::get_neon_intrinsics;

pub struct ArmArchitectureTest {
    intrinsics: Vec<Intrinsic<ArmIntrinsicType>>,
}

impl SupportedArchitectureTest for ArmArchitectureTest {
    type IntrinsicImpl = ArmIntrinsicType;

    fn intrinsics(&self) -> &[Intrinsic<ArmIntrinsicType>] {
        &self.intrinsics
    }

    const NOTICE: &str = config::NOTICE;

    const PLATFORM_C_HEADERS: &[&str] = &["arm_neon.h", "arm_acle.h", "arm_fp16.h"];

    const PLATFORM_RUST_DEFINITIONS: &str = config::PLATFORM_RUST_DEFINITIONS;
    const PLATFORM_RUST_CFGS: &str = config::PLATFORM_RUST_CFGS;

    fn arch_flags(&self) -> Vec<&str> {
        vec!["-march=armv8.6a+crypto+crc+dotprod+fp16"]
    }

    fn create(cli_options: ProcessedCli) -> Self {
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
            // Skip pointers for now, we would probably need to look at the return
            // type to work out how many elements we need to point to.
            .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
            // Skip intrinsics with 128-bit elements (e.g. `p128`)
            .filter(|i| !i.arguments.iter().any(|a| a.ty.inner_size() == 128))
            // Skip intrinsics from `--skip`
            .filter(|i| !cli_options.skip.contains(&i.name))
            // Skip A64-specific intrinsics on A32
            .filter(|i| !(a32 && i.arch_tags == vec!["A64".to_string()]))
            .take(sample_size)
            .collect::<Vec<_>>();

        Self { intrinsics }
    }
}
