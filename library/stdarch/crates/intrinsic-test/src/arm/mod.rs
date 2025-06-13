mod compile;
mod config;
mod intrinsic;
mod json_parser;
mod types;

use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::compare::compare_outputs;
use crate::common::gen_rust::compile_rust_programs;
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
use crate::common::intrinsic_helpers::TypeKind;
use crate::common::write_file::{write_c_testfiles, write_rust_testfiles};
use compile::compile_c_arm;
use config::{AARCH_CONFIGURATIONS, F16_FORMATTING_DEF, POLY128_OSTREAM_DEF, build_notices};
use intrinsic::ArmIntrinsicType;
use json_parser::get_neon_intrinsics;

pub struct ArmArchitectureTest {
    intrinsics: Vec<Intrinsic<ArmIntrinsicType>>,
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
            intrinsics,
            cli_options,
        })
    }

    fn build_c_file(&self) -> bool {
        let compiler = self.cli_options.cpp_compiler.as_deref();
        let target = &self.cli_options.target;
        let cxx_toolchain_dir = self.cli_options.cxx_toolchain_dir.as_deref();
        let c_target = "aarch64";

        let intrinsics_name_list = write_c_testfiles(
            &self
                .intrinsics
                .iter()
                .map(|i| i as &dyn IntrinsicDefinition<_>)
                .collect::<Vec<_>>(),
            target,
            c_target,
            &["arm_neon.h", "arm_acle.h", "arm_fp16.h"],
            &build_notices("// "),
            &[POLY128_OSTREAM_DEF],
        );

        match compiler {
            None => true,
            Some(compiler) => compile_c_arm(
                intrinsics_name_list.as_slice(),
                compiler,
                target,
                cxx_toolchain_dir,
            ),
        }
    }

    fn build_rust_file(&self) -> bool {
        let rust_target = if self.cli_options.target.contains("v7") {
            "arm"
        } else {
            "aarch64"
        };
        let target = &self.cli_options.target;
        let toolchain = self.cli_options.toolchain.as_deref();
        let linker = self.cli_options.linker.as_deref();
        let intrinsics_name_list = write_rust_testfiles(
            self.intrinsics
                .iter()
                .map(|i| i as &dyn IntrinsicDefinition<_>)
                .collect::<Vec<_>>(),
            rust_target,
            &build_notices("// "),
            F16_FORMATTING_DEF,
            AARCH_CONFIGURATIONS,
        );

        compile_rust_programs(intrinsics_name_list, toolchain, target, linker)
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
