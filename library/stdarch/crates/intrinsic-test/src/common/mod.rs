use std::{fs::File, io};

use rayon::prelude::*;

use cli::ProcessedCli;

use crate::common::{
    gen_c::write_wrapper_c,
    gen_rust::{write_bin_cargo_toml, write_lib_cargo_toml, write_lib_rs},
    intrinsic::Intrinsic,
    intrinsic_helpers::IntrinsicTypeDefinition,
};

pub mod argument;
pub mod cli;
pub mod constraint;
pub mod gen_c;
pub mod gen_rust;
pub mod indentation;
pub mod intrinsic;
pub mod intrinsic_helpers;
pub mod values;

/// Architectures must support this trait
/// to be successfully tested.
pub trait SupportedArchitectureTest {
    type IntrinsicImpl: IntrinsicTypeDefinition + Sync;

    fn cli_options(&self) -> &ProcessedCli;
    fn intrinsics(&self) -> &[Intrinsic<Self::IntrinsicImpl>];

    fn create(cli_options: ProcessedCli) -> Self;

    const NOTICE: &str;

    const PLATFORM_C_HEADERS: &[&str];

    const PLATFORM_RUST_CFGS: &str;
    const PLATFORM_RUST_DEFINITIONS: &str;

    fn generate_c_file(&self) {
        let (chunk_size, chunk_count) = manual_chunk(self.intrinsics().len(), 400);

        std::fs::create_dir_all("c_programs").unwrap();
        self.intrinsics()
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let c_filename = format!("c_programs/wrapper_{i}.cpp");
                let mut file = File::create(&c_filename).unwrap();
                write_wrapper_c(&mut file, Self::NOTICE, Self::PLATFORM_C_HEADERS, chunk)
            })
            .collect::<io::Result<()>>()
            .unwrap();
    }

    fn generate_rust_file(&self) {
        std::fs::create_dir_all("rust_programs/src").unwrap();

        let (chunk_size, chunk_count) = manual_chunk(self.intrinsics().len(), 400);

        let mut cargo = File::create("rust_programs/Cargo.toml").unwrap();
        write_bin_cargo_toml(&mut cargo, chunk_count).unwrap();

        self.intrinsics()
            .par_chunks(chunk_size)
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
    }
}

// pub fn chunk_info(intrinsic_count: usize) -> (usize, usize) {
//     let available_parallelism = std::thread::available_parallelism().unwrap().get();
//     let chunk_size = intrinsic_count.div_ceil(Ord::min(available_parallelism, intrinsic_count));

//     (chunk_size, intrinsic_count.div_ceil(chunk_size))
// }

pub fn manual_chunk(intrinsic_count: usize, chunk_size: usize) -> (usize, usize) {
    (chunk_size, intrinsic_count.div_ceil(chunk_size))
}
