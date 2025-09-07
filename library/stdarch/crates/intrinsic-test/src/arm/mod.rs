mod argument;
mod compile;
mod config;
mod intrinsic;
mod json_parser;
mod types;

use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::compile_c::CppCompilation;
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use intrinsic::ArmIntrinsicType;
use json_parser::get_neon_intrinsics;

pub struct ArmArchitectureTest {
    intrinsics: Vec<Intrinsic<ArmIntrinsicType>>,
    cli_options: ProcessedCli,
}

impl SupportedArchitectureTest for ArmArchitectureTest {
    type IntrinsicImpl = ArmIntrinsicType;

    fn cli_options(&self) -> &ProcessedCli {
        &self.cli_options
    }

    fn intrinsics(&self) -> &[Intrinsic<ArmIntrinsicType>] {
        &self.intrinsics
    }

    const NOTICE: &str = config::NOTICE;

    const PLATFORM_C_HEADERS: &[&str] = &["arm_neon.h", "arm_acle.h", "arm_fp16.h"];
    const PLATFORM_C_DEFINITIONS: &str = config::POLY128_OSTREAM_DEF;
    const PLATFORM_C_FORWARD_DECLARATIONS: &str = config::POLY128_OSTREAM_DECL;

    const PLATFORM_RUST_DEFINITIONS: &str = config::F16_FORMATTING_DEF;
    const PLATFORM_RUST_CFGS: &str = config::AARCH_CONFIGURATIONS;

    fn cpp_compilation(&self) -> Option<CppCompilation> {
        compile::build_cpp_compilation(&self.cli_options)
    }

    fn create(cli_options: ProcessedCli) -> Self {
        let a32 = cli_options.target.contains("v7");
        let mut intrinsics = get_neon_intrinsics(&cli_options.filename, &cli_options.target)
            .expect("Error parsing input file");

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));

        let mut intrinsics = intrinsics
            .into_iter()
            // Not sure how we would compare intrinsic that returns void.
            .filter(|i| i.results.kind() != TypeKind::Void)
            .filter(|i| i.results.kind() != TypeKind::BFloat)
            .filter(|i| !i.arguments.iter().any(|a| a.ty.kind() == TypeKind::BFloat))
            // Skip pointers for now, we would probably need to look at the return
            // type to work out how many elements we need to point to.
            .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
            .filter(|i| !i.arguments.iter().any(|a| a.ty.inner_size() == 128))
            .filter(|i| !cli_options.skip.contains(&i.name))
            .filter(|i| !(a32 && i.arch_tags == vec!["A64".to_string()]))
            .collect::<Vec<_>>();
        intrinsics.dedup();

        Self {
            intrinsics,
            cli_options,
        }
    }
}
