use super::gen_rust::{create_rust_test_program, setup_rust_file_paths};
use super::intrinsic::IntrinsicDefinition;
use super::intrinsic_helpers::IntrinsicTypeDefinition;
use std::fs::File;
use std::io::Write;

pub fn write_file(filename: &String, code: String) {
    let mut file = File::create(filename).unwrap();
    file.write_all(code.into_bytes().as_slice()).unwrap();
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
    let filename_mapping = setup_rust_file_paths(&intrinsics_name_list);

    intrinsics.iter().for_each(|&i| {
        let rust_code = create_rust_test_program(i, rust_target, notice, definitions, cfg);
        if let Some(filename) = filename_mapping.get(&i.name()) {
            write_file(filename, rust_code)
        }
    });

    intrinsics_name_list
}
