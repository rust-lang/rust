mod compile;
mod config;
mod constraint;
mod intrinsic;
mod types;
mod xml_parser;

use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::compile_c::CppCompilation;
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use intrinsic::X86IntrinsicType;
use itertools::Itertools;
use xml_parser::get_xml_intrinsics;

pub struct X86ArchitectureTest {
    intrinsics: Vec<Intrinsic<X86IntrinsicType>>,
    cli_options: ProcessedCli,
}

impl SupportedArchitectureTest for X86ArchitectureTest {
    type IntrinsicImpl = X86IntrinsicType;

    fn cli_options(&self) -> &ProcessedCli {
        &self.cli_options
    }

    fn intrinsics(&self) -> &[Intrinsic<X86IntrinsicType>] {
        &self.intrinsics
    }

    fn cpp_compilation(&self) -> Option<CppCompilation> {
        compile::build_cpp_compilation(&self.cli_options)
    }

    const NOTICE: &str = config::NOTICE;

    const PLATFORM_C_HEADERS: &[&str] = &["immintrin.h", "cstddef", "cstdint"];
    const PLATFORM_C_DEFINITIONS: &str = config::PLATFORM_C_DEFINITIONS;
    const PLATFORM_C_FORWARD_DECLARATIONS: &str = config::PLATFORM_C_FORWARD_DECLARATIONS;

    const PLATFORM_RUST_DEFINITIONS: &str = config::PLATFORM_RUST_DEFINITIONS;
    const PLATFORM_RUST_CFGS: &str = config::PLATFORM_RUST_CFGS;

    fn create(cli_options: ProcessedCli) -> Self {
        let intrinsics =
            get_xml_intrinsics(&cli_options.filename).expect("Error parsing input file");

        let sample_percentage: usize = cli_options.sample_percentage as usize;

        let mut intrinsics = intrinsics
            .into_iter()
            // Not sure how we would compare intrinsic that returns void.
            .filter(|i| i.results.kind() != TypeKind::Void)
            .filter(|i| i.results.kind() != TypeKind::BFloat)
            .filter(|i| i.arguments.args.len() > 0)
            .filter(|i| !i.arguments.iter().any(|a| a.ty.kind() == TypeKind::BFloat))
            // Skip pointers for now, we would probably need to look at the return
            // type to work out how many elements we need to point to.
            .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
            .filter(|i| !i.arguments.iter().any(|a| a.ty.inner_size() == 128))
            .filter(|i| !cli_options.skip.contains(&i.name))
            .unique_by(|i| i.name.clone())
            .collect::<Vec<_>>();

        let sample_size = (intrinsics.len() * sample_percentage) / 100;
        intrinsics.truncate(sample_size);

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));
        Self {
            intrinsics: intrinsics,
            cli_options: cli_options,
        }
    }
}
