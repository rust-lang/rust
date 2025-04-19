mod config;
mod constraint;
mod functions;
mod intrinsic;
mod json_parser;
mod types;

use crate::arm::constraint::Constraint;
use crate::arm::intrinsic::ArmIntrinsicType;
use crate::common::SupportedArchitectureTest;
use crate::common::compare::compare_outputs;
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_types::{BaseIntrinsicTypeDefinition, TypeKind};
use crate::common::types::ProcessedCli;
use functions::{build_c, build_rust};
use json_parser::get_neon_intrinsics;

pub struct ArmArchitectureTest {
    intrinsics: Vec<Intrinsic<ArmIntrinsicType, Constraint>>,
    cli_options: ProcessedCli,
}

impl SupportedArchitectureTest for ArmArchitectureTest {
    fn create(cli_options: ProcessedCli) -> Box<Self> {
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

        Box::new(Self {
            intrinsics: intrinsics,
            cli_options: cli_options,
        })
    }

    fn build_c_file(&self) -> bool {
        build_c(
            &self.intrinsics,
            self.cli_options.cpp_compiler.as_deref(),
            &self.cli_options.target,
            self.cli_options.cxx_toolchain_dir.as_deref(),
        )
    }

    fn build_rust_file(&self) -> bool {
        build_rust(
            &self.intrinsics,
            self.cli_options.toolchain.as_deref(),
            &self.cli_options.target,
            self.cli_options.linker.as_deref(),
        )
    }

    fn compare_outputs(&self) -> bool {
        if let Some(ref toolchain) = self.cli_options.toolchain {
            let intrinsics_name_list = self
                .intrinsics
                .iter()
                .map(|i| i.name.clone())
                .collect::<Vec<_>>();

            compare_outputs(
                &intrinsics_name_list,
                toolchain,
                &self.cli_options.c_runner,
                &self.cli_options.target,
            )
        } else {
            true
        }
    }
}
