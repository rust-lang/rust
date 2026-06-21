use std::{fs::File, io};

use rayon::prelude::*;

use cli::ProcessedCli;

use crate::common::{
    gen_c::write_wrapper_c,
    gen_rust::{
        run_rustfmt, write_bin_cargo_toml, write_build_rs, write_lib_cargo_toml, write_lib_rs,
    },
    intrinsic::Intrinsic,
    intrinsic_helpers::TypeDefinition,
};

pub mod argument;
pub mod cli;
pub mod constraint;
pub mod intrinsic;
pub mod intrinsic_helpers;

mod gen_c;
mod gen_rust;
mod values;

// The number of times each intrinsic will be called - influences the generation of the
// test arrays to minimise repeated testing of the same test values.
pub(crate) const PASSES: u32 = 20;

/// Architectures must support this trait
/// to be successfully tested.
pub trait SupportedArchitecture: Sized {
    type Type: TypeDefinition + std::fmt::Debug + PartialEq + Sync;

    fn intrinsics(&self) -> &[Intrinsic<Self>];

    fn create(cli_options: &ProcessedCli) -> Self;

    const NOTICE: &str;

    const C_PRELUDE: &str;
    const RUST_PRELUDE: &str;

    fn c_compiler_flags(&self, cli_options: &ProcessedCli) -> Vec<&str>;

    fn generate_c_file(&self) {
        let (max_chunk_size, _chunk_count) = manual_chunk(self.intrinsics().len());

        std::fs::create_dir_all("c_programs").unwrap();
        self.intrinsics()
            .par_chunks(max_chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let c_filename = format!("c_programs/wrapper_{i}.c");
                let mut file = File::create(&c_filename).unwrap();
                write_wrapper_c(&mut file, chunk)
            })
            .collect::<io::Result<()>>()
            .unwrap();
    }

    fn generate_rust_file(&self, cli_options: &ProcessedCli) {
        let arch_flags = self.c_compiler_flags(cli_options);

        std::fs::create_dir_all("rust_programs").unwrap();

        let (max_chunk_size, chunk_count) = manual_chunk(self.intrinsics().len());

        let mut cargo = File::create("rust_programs/Cargo.toml").unwrap();
        write_bin_cargo_toml(&mut cargo, chunk_count).unwrap();

        self.intrinsics()
            .chunks(max_chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                std::fs::create_dir_all(format!("rust_programs/mod_{i}/src"))?;

                let rust_filename = format!("rust_programs/mod_{i}/src/lib.rs");
                trace!("generating `{rust_filename}`");
                let mut file = File::create(&rust_filename)?;

                write_lib_rs(&mut file, i, chunk)?;
                run_rustfmt(&rust_filename);

                let toml_filename = format!("rust_programs/mod_{i}/Cargo.toml");
                trace!("generating `{toml_filename}`");
                let mut file = File::create(toml_filename).unwrap();

                write_lib_cargo_toml(&mut file, &format!("mod_{i}"))?;

                let build_rs_filename = format!("rust_programs/mod_{i}/build.rs");
                trace!("generating `{build_rs_filename}`");
                let mut file = File::create(&build_rs_filename).unwrap();

                write_build_rs(&mut file, i, &arch_flags, &cli_options).unwrap();
                run_rustfmt(&build_rs_filename);

                Ok(())
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();
    }
}

pub fn manual_chunk(intrinsic_count: usize) -> (usize, usize) {
    let ncores = std::thread::available_parallelism().unwrap().into();
    let max_intrinsics_per_chunk = intrinsic_count.div_ceil(ncores);
    let number_of_chunks = intrinsic_count.div_ceil(max_intrinsics_per_chunk);
    (max_intrinsics_per_chunk, number_of_chunks)
}
