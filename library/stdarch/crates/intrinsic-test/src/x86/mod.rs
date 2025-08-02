mod intrinsic;
mod types;
mod xml_parser;

use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::intrinsic::Intrinsic;
use intrinsic::X86IntrinsicType;

pub struct X86ArchitectureTest {
    intrinsics: Vec<Intrinsic<X86IntrinsicType>>,
    cli_options: ProcessedCli,
}

impl SupportedArchitectureTest for X86ArchitectureTest {
    fn create(cli_options: ProcessedCli) -> Box<Self> {
        todo!("create in X86ArchitectureTest is not implemented")
    }

    fn build_c_file(&self) -> bool {
        todo!("build_c_file in X86ArchitectureTest is not implemented")
    }

    fn build_rust_file(&self) -> bool {
        todo!("build_rust_file in X86ArchitectureTest is not implemented")
    }

    fn compare_outputs(&self) -> bool {
        todo!("compare_outputs in X86ArchitectureTest is not implemented")
    }
}