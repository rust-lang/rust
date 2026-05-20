use std::{fs::File, io};

use rayon::prelude::*;

use cli::ProcessedCli;

use crate::common::{
    gen_c::write_wrapper_c,
    gen_rust::{write_bin_cargo_toml, write_build_rs, write_lib_cargo_toml, write_lib_rs},
    intrinsic::Intrinsic,
    intrinsic_helpers::IntrinsicTypeDefinition,
};

pub mod argument;
pub mod cli;
pub mod constraint;
pub mod intrinsic;
pub mod intrinsic_helpers;

mod gen_c;
mod gen_rust;
mod indentation;
mod values;

/// Architectures must support this trait
/// to be successfully tested.
pub trait SupportedArchitectureTest {
    type IntrinsicImpl: IntrinsicTypeDefinition + Sync;

    fn intrinsics(&self) -> &[Intrinsic<Self::IntrinsicImpl>];

    fn create(cli_options: ProcessedCli) -> Self;

    const NOTICE: &str;

    const PLATFORM_C_HEADERS: &[&str];

    const PLATFORM_RUST_CFGS: &str;
    const PLATFORM_RUST_DEFINITIONS: &str;

    fn arch_flags(&self) -> Vec<&str>;

    fn generate_c_file(&self) {
        let (chunk_size, _chunk_count) = manual_chunk(self.intrinsics().len());

        std::fs::create_dir_all("c_programs").unwrap();
        self.intrinsics()
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let c_filename = format!("c_programs/wrapper_{i}.c");
                let mut file = File::create(&c_filename).unwrap();
                write_wrapper_c(&mut file, Self::NOTICE, Self::PLATFORM_C_HEADERS, chunk)
            })
            .collect::<io::Result<()>>()
            .unwrap();
    }

    fn generate_rust_file(&self) {
        let arch_flags = self.arch_flags();

        std::fs::create_dir_all("rust_programs").unwrap();

        let (chunk_size, chunk_count) = manual_chunk(self.intrinsics().len());

        let mut cargo = File::create("rust_programs/Cargo.toml").unwrap();
        write_bin_cargo_toml(&mut cargo, chunk_count).unwrap();

        self.intrinsics()
            .chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                std::fs::create_dir_all(format!("rust_programs/mod_{i}/src"))?;

                let rust_filename = format!("rust_programs/mod_{i}/src/lib.rs");
                trace!("generating `{rust_filename}`");
                let mut file = File::create(rust_filename)?;

                write_lib_rs(
                    &mut file,
                    Self::NOTICE,
                    Self::PLATFORM_RUST_CFGS,
                    Self::PLATFORM_RUST_DEFINITIONS,
                    i,
                    chunk,
                )?;

                let toml_filename = format!("rust_programs/mod_{i}/Cargo.toml");
                trace!("generating `{toml_filename}`");
                let mut file = File::create(toml_filename).unwrap();

                write_lib_cargo_toml(&mut file, &format!("mod_{i}"))?;

                let build_rs_filename = format!("rust_programs/mod_{i}/build.rs");
                trace!("generating `{build_rs_filename}`");
                let mut file = File::create(build_rs_filename).unwrap();

                write_build_rs(&mut file, i, &arch_flags).unwrap();

                Ok(())
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();
    }
}

pub fn manual_chunk(intrinsic_count: usize) -> (usize, usize) {
    let ncores = std::thread::available_parallelism().unwrap().into();
    (intrinsic_count.div_ceil(ncores), ncores)
}
