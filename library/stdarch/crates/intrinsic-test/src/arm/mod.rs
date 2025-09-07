mod argument;
mod compile;
mod config;
mod intrinsic;
mod json_parser;
mod types;

use std::fs::File;

use rayon::prelude::*;

use crate::common::cli::ProcessedCli;
use crate::common::compile_c::CppCompilation;
use crate::common::gen_c::{write_main_cpp, write_mod_cpp};
use crate::common::gen_rust::{
    compile_rust_programs, write_bin_cargo_toml, write_lib_cargo_toml, write_lib_rs, write_main_rs,
};
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use crate::common::{SupportedArchitectureTest, chunk_info};
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

    const NOTICE: &str = config::NOTICE;

    const PLATFORM_C_HEADERS: &[&str] = &["arm_neon.h", "arm_acle.h", "arm_fp16.h"];
    const PLATFORM_C_DEFINITIONS: &str = config::POLY128_OSTREAM_DEF;

    const PLATFORM_RUST_DEFINITIONS: &str = config::F16_FORMATTING_DEF;
    const PLATFORM_RUST_CFGS: &str = config::AARCH_CONFIGURATIONS;

    fn cpp_compilation(&self) -> Option<CppCompilation> {
        compile::build_cpp_compilation(&self.cli_options)
    }

    fn build_c_file(&self) -> bool {
        let c_target = "aarch64";

        let (chunk_size, chunk_count) = chunk_info(self.intrinsics().len());

        let cpp_compiler_wrapped = self.cpp_compilation();

        std::fs::create_dir_all("c_programs").unwrap();
        self.intrinsics()
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let c_filename = format!("c_programs/mod_{i}.cpp");
                let mut file = File::create(&c_filename).unwrap();
                write_mod_cpp(
                    &mut file,
                    Self::NOTICE,
                    c_target,
                    Self::PLATFORM_C_HEADERS,
                    chunk,
                )
                .unwrap();

                // compile this cpp file into a .o file.
                //
                // This is done because `cpp_compiler_wrapped` is None when
                // the --generate-only flag is passed
                if let Some(cpp_compiler) = cpp_compiler_wrapped.as_ref() {
                    let output = cpp_compiler
                        .compile_object_file(&format!("mod_{i}.cpp"), &format!("mod_{i}.o"))?;
                    assert!(output.status.success(), "{output:?}");
                }

                Ok(())
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();

        let mut file = File::create("c_programs/main.cpp").unwrap();
        write_main_cpp(
            &mut file,
            c_target,
            Self::PLATFORM_C_DEFINITIONS,
            self.intrinsics().iter().map(|i| i.name.as_str()),
        )
        .unwrap();

        // This is done because `cpp_compiler_wrapped` is None when
        // the --generate-only flag is passed
        if let Some(cpp_compiler) = cpp_compiler_wrapped.as_ref() {
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
        }

        true
    }

    fn build_rust_file(&self) -> bool {
        std::fs::create_dir_all("rust_programs/src").unwrap();

        let architecture = if self.cli_options.target.contains("v7") {
            "arm"
        } else {
            "aarch64"
        };

        let (chunk_size, chunk_count) = chunk_info(self.intrinsics.len());

        let mut cargo = File::create("rust_programs/Cargo.toml").unwrap();
        write_bin_cargo_toml(&mut cargo, chunk_count).unwrap();

        let mut main_rs = File::create("rust_programs/src/main.rs").unwrap();
        write_main_rs(
            &mut main_rs,
            chunk_count,
            Self::PLATFORM_RUST_CFGS,
            "",
            self.intrinsics.iter().map(|i| i.name.as_str()),
        )
        .unwrap();

        let target = &self.cli_options.target;
        let toolchain = self.cli_options.toolchain.as_deref();
        let linker = self.cli_options.linker.as_deref();

        self.intrinsics
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                std::fs::create_dir_all(format!("rust_programs/mod_{i}/src"))?;

                let rust_filename = format!("rust_programs/mod_{i}/src/lib.rs");
                trace!("generating `{rust_filename}`");
                let mut file = File::create(rust_filename)?;

                write_lib_rs(
                    &mut file,
                    architecture,
                    Self::NOTICE,
                    Self::PLATFORM_RUST_CFGS,
                    Self::PLATFORM_RUST_DEFINITIONS,
                    chunk,
                )?;

                let toml_filename = format!("rust_programs/mod_{i}/Cargo.toml");
                trace!("generating `{toml_filename}`");
                let mut file = File::create(toml_filename).unwrap();

                write_lib_cargo_toml(&mut file, &format!("mod_{i}"))?;

                Ok(())
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();

        compile_rust_programs(toolchain, target, linker)
    }
}
