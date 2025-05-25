use super::gen_c::gen_c_program;
use super::gen_rust::{create_rust_filenames, gen_rust_program};
use super::intrinsic_helpers::IntrinsicTypeDefinition;
use crate::common::gen_c::create_c_filenames;
use crate::common::intrinsic::IntrinsicDefinition;
use crate::common::write_file;

pub fn write_c_testfiles<T: IntrinsicTypeDefinition + Sized>(
    intrinsics: &Vec<&dyn IntrinsicDefinition<T>>,
    target: &str,
    c_target: &str,
    headers: &[&str],
    notice: &str,
    arch_specific_definitions: &[&str],
) -> Vec<String> {
    let intrinsics_name_list = intrinsics
        .iter()
        .map(|i| i.name().clone())
        .collect::<Vec<_>>();
    let filename_mapping = create_c_filenames(&intrinsics_name_list);

    intrinsics.iter().for_each(|&i| {
        let c_code = gen_c_program(
            i,
            headers,
            target,
            c_target,
            notice,
            arch_specific_definitions,
        );
        match filename_mapping.get(&i.name()) {
            Some(filename) => write_file(filename, c_code),
            None => {}
        };
    });

    intrinsics_name_list
}

pub fn write_rust_testfiles<T: IntrinsicTypeDefinition>(
    intrinsics: Vec<&dyn IntrinsicDefinition<T>>,
    rust_target: &str,
    notice: &str,
    definitions: &str,
    cfg: &str,
) -> Vec<String> {
    let intrinsics_name_list = intrinsics
        .iter()
        .map(|i| i.name().clone())
        .collect::<Vec<_>>();
    let filename_mapping = create_rust_filenames(&intrinsics_name_list);

    intrinsics.iter().for_each(|&i| {
        let rust_code = gen_rust_program(i, rust_target, notice, definitions, cfg);
        match filename_mapping.get(&i.name()) {
            Some(filename) => write_file(filename, rust_code),
            None => {}
        }
    });

    intrinsics_name_list
}
