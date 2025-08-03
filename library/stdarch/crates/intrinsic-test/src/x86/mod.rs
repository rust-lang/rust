mod constraint;
mod intrinsic;
mod types;
mod xml_parser;

use crate::common::cli::ProcessedCli;
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
use crate::common::intrinsic_helpers::TypeKind;
use crate::common::SupportedArchitectureTest;
use intrinsic::X86IntrinsicType;
use xml_parser::get_xml_intrinsics;

pub struct X86ArchitectureTest {
    intrinsics: Vec<Intrinsic<X86IntrinsicType>>,
    cli_options: ProcessedCli,
}

impl SupportedArchitectureTest for X86ArchitectureTest {
    fn create(cli_options: ProcessedCli) -> Box<Self> {
        let intrinsics =
            get_xml_intrinsics(&cli_options.filename).expect("Error parsing input file");

        let mut intrinsics = intrinsics
            .into_iter()
            // Not sure how we would compare intrinsic that returns void.
            .filter(|i| i.results.kind() != TypeKind::Void)
            .filter(|i| i.results.kind() != TypeKind::BFloat)
            .filter(|i| i.arguments().args.len() > 0)
            .filter(|i| !i.arguments.iter().any(|a| a.ty.kind() == TypeKind::BFloat))
            // Skip pointers for now, we would probably need to look at the return
            // type to work out how many elements we need to point to.
            .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
            .filter(|i| !i.arguments.iter().any(|a| a.ty.inner_size() == 128))
            .filter(|i| !cli_options.skip.contains(&i.name))
            .collect::<Vec<_>>();

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));
        Box::new(Self {
            intrinsics: intrinsics,
            cli_options: cli_options,
        })
    }

    fn build_c_file(&self) -> bool {
        todo!("build_c_file in X86ArchitectureTest is not implemented")
    }

    fn build_rust_file(&self) -> bool {
        todo!("build_rust_file in X86ArchitectureTest is not implemented")
    }

    fn compare_outputs(&self) -> bool {
        todo!("compare_outputs in X86ArchitectureTest is not implemented")
    }
}