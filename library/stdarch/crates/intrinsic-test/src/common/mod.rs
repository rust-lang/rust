use cli::ProcessedCli;

use crate::common::{intrinsic::Intrinsic, intrinsic_helpers::IntrinsicTypeDefinition};

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

/// Architectures must support this trait
/// to be successfully tested.
pub trait SupportedArchitectureTest {
    type IntrinsicImpl: IntrinsicTypeDefinition;

    fn cli_options(&self) -> &ProcessedCli;
    fn intrinsics(&self) -> &[Intrinsic<Self::IntrinsicImpl>];

    fn create(cli_options: ProcessedCli) -> Self;

    const PLATFORM_HEADERS: &[&str];

    fn build_c_file(&self) -> bool;
    fn build_rust_file(&self) -> bool;

    fn compare_outputs(&self) -> bool {
        if self.cli_options().toolchain.is_some() {
            let intrinsics_name_list = self
                .intrinsics()
                .iter()
                .map(|i| i.name.clone())
                .collect::<Vec<_>>();

            compare::compare_outputs(
                &intrinsics_name_list,
                &self.cli_options().runner,
                &self.cli_options().target,
            )
        } else {
            true
        }
    }
}

pub fn chunk_info(intrinsic_count: usize) -> (usize, usize) {
    let available_parallelism = std::thread::available_parallelism().unwrap().get();
    let chunk_size = intrinsic_count.div_ceil(Ord::min(available_parallelism, intrinsic_count));

    (chunk_size, intrinsic_count.div_ceil(chunk_size))
}
