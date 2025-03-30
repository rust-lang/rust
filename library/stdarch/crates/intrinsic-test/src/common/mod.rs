use crate::common::types::ProcessedCli;
use std::fs::File;
use std::io::Write;

pub mod compare;
pub mod gen_c;
pub mod gen_rust;
pub mod types;
pub mod values;

/// Architectures must support this trait
/// to be successfully tested.
pub trait SupportedArchitectureTest {
    fn create(cli_options: ProcessedCli) -> Self;
    fn build_c_file(&self) -> bool;
    fn build_rust_file(&self) -> bool;
    fn compare_outputs(&self) -> bool;
}

pub fn write_file(filename: &String, code: String) {
    let mut file = File::create(&filename).unwrap();
    file.write_all(code.into_bytes().as_slice()).unwrap();
}
