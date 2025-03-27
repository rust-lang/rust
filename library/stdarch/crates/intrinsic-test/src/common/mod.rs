use crate::common::types::ProcessedCli;

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
