mod compile;
mod config;
mod intrinsic;
mod json_parser;
mod types;

use std::fs::File;

use rayon::prelude::*;

use crate::arm::config::POLY128_OSTREAM_DEF;
use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::compare::compare_outputs;
use crate::common::gen_c::{write_main_cpp, write_mod_cpp};
use crate::common::gen_rust::compile_rust_programs;
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
use crate::common::intrinsic_helpers::TypeKind;
use crate::common::write_file::write_rust_testfiles;
use config::{AARCH_CONFIGURATIONS, F16_FORMATTING_DEF, build_notices};
use intrinsic::ArmIntrinsicType;
use json_parser::get_neon_intrinsics;

pub struct ArmArchitectureTest {
    intrinsics: Vec<Intrinsic<ArmIntrinsicType>>,
    cli_options: ProcessedCli,
}

fn chunk_info(intrinsic_count: usize) -> (usize, usize) {
    let available_parallelism = std::thread::available_parallelism().unwrap().get();
    let chunk_size = intrinsic_count.div_ceil(Ord::min(available_parallelism, intrinsic_count));

    (chunk_size, intrinsic_count.div_ceil(chunk_size))
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
        let c_target = "aarch64";
        let platform_headers = &["arm_neon.h", "arm_acle.h", "arm_fp16.h"];

        let (chunk_size, chunk_count) = chunk_info(self.intrinsics.len());

        let cpp_compiler = compile::build_cpp_compilation(&self.cli_options).unwrap();

        let notice = &build_notices("// ");
        self.intrinsics
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let c_filename = format!("c_programs/mod_{i}.cpp");
                let mut file = File::create(&c_filename).unwrap();
                write_mod_cpp(&mut file, notice, c_target, platform_headers, chunk).unwrap();

                // compile this cpp file into a .o file
                let output = cpp_compiler
                    .compile_object_file(&format!("mod_{i}.cpp"), &format!("mod_{i}.o"))?;
                assert!(output.status.success(), "{output:?}");

                Ok(())
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();

        let mut file = File::create("c_programs/main.cpp").unwrap();
        write_main_cpp(
            &mut file,
            c_target,
            POLY128_OSTREAM_DEF,
            self.intrinsics.iter().map(|i| i.name.as_str()),
        )
        .unwrap();

        // compile this cpp file into a .o file
        info!("compiling main.cpp");
        let output = cpp_compiler
            .compile_object_file("main.cpp", "intrinsic-test-programs.o")
            .unwrap();
        assert!(output.status.success(), "{output:?}");

        let object_files = (0..chunk_count)
            .map(|i| format!("mod_{i}.o"))
            .chain(["intrinsic-test-programs.o".to_owned()]);

        let output = cpp_compiler
            .link_executable(object_files, "intrinsic-test-programs")
            .unwrap();
        assert!(output.status.success(), "{output:?}");

        true
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
        if self.cli_options.toolchain.is_some() {
            let intrinsics_name_list = self
                .intrinsics
                .iter()
                .map(|i| i.name.clone())
                .collect::<Vec<_>>();

            compare_outputs(
                &intrinsics_name_list,
                &self.cli_options.runner,
                &self.cli_options.target,
            )
        } else {
            true
        }
    }
}
