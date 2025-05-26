use cli::ProcessedCli;

pub mod argument;
pub mod cli;
pub mod compare;
pub mod compile_c;
pub mod constraint;
pub mod gen_c;
pub mod gen_rust;
pub mod indentation;
pub mod intrinsic;
pub mod intrinsic_helpers;
pub mod values;
pub mod write_file;

/// Architectures must support this trait
/// to be successfully tested.
pub trait SupportedArchitectureTest {
    fn create(cli_options: ProcessedCli) -> Box<Self>
    where
        Self: Sized;
    fn build_c_file(&self) -> bool;
    fn build_rust_file(&self) -> bool;
    fn compare_outputs(&self) -> bool;
}
