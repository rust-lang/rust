mod compile;
mod config;
mod constraint;
mod intrinsic;
mod types;
mod xml_parser;

use std::fs::{self, File};
use rayon::prelude::*;

use crate::common::cli::ProcessedCli;
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
use crate::common::intrinsic_helpers::TypeKind;
use crate::common::{SupportedArchitectureTest, chunk_info};
use crate::common::gen_c::{write_main_cpp, write_mod_cpp};
use intrinsic::X86IntrinsicType;
use xml_parser::get_xml_intrinsics;
use config::build_notices;

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
        let c_target = "x86_64";
        let platform_headers = &["immintrin.h"];

        let (chunk_size, chunk_count) = chunk_info(self.intrinsics.len());

        let cpp_compiler_wrapped = compile::build_cpp_compilation(&self.cli_options);

        let notice = &build_notices("// ");
        fs::create_dir_all("c_programs").unwrap();
        self.intrinsics
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let c_filename = format!("c_programs/mod_{i}.cpp");
                let mut file = File::create(&c_filename).unwrap();
                write_mod_cpp(&mut file, notice, c_target, platform_headers, chunk).unwrap();

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
            "\n",
            self.intrinsics.iter().map(|i| i.name.as_str()),
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
        todo!("build_rust_file in X86ArchitectureTest is not implemented")
    }

    fn compare_outputs(&self) -> bool {
        todo!("compare_outputs in X86ArchitectureTest is not implemented")
    }
}