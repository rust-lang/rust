#[macro_use]
extern crate log;

mod arm;
mod common;
mod x86;

use arm::ArmArchitectureTest;
use common::SupportedArchitectureTest;
use common::cli::{Cli, ProcessedCli};
use x86::X86ArchitectureTest;

fn main() {
    pretty_env_logger::init();
    let args: Cli = clap::Parser::parse();
    let processed_cli_options = ProcessedCli::new(args);

    if processed_cli_options.target.starts_with("arm")
        | processed_cli_options.target.starts_with("aarch64")
    {
        run(ArmArchitectureTest::create(processed_cli_options))
    } else if processed_cli_options.target.starts_with("x86") {
        run(X86ArchitectureTest::create(processed_cli_options))
    } else {
        unimplemented!("Unsupported target {}", processed_cli_options.target)
    }
}

fn run(test_environment: impl SupportedArchitectureTest) {
    info!("building C binaries");
    test_environment.generate_c_file();

    info!("building Rust binaries");
    test_environment.generate_rust_file();
}
