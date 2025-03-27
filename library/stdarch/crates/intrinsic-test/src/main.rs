#![feature(slice_partition_dedup)]
#[macro_use]
extern crate log;

mod arm;
mod common;

use arm::ArmTestProcessor;
use common::SupportedArchitectureTest;
use common::types::{Cli, ProcessedCli};

fn main() {
    pretty_env_logger::init();
    let args: Cli = clap::Parser::parse();
    let processed_cli_options = ProcessedCli::new(args);

    // TODO: put this in a match block to support more architectures
    let test_environment = ArmTestProcessor::create(processed_cli_options);

    if !test_environment.build_c_file() {
        std::process::exit(2);
    }
    if !test_environment.build_rust_file() {
        std::process::exit(3);
    }
    if !test_environment.compare_outputs() {
        std::process::exit(1);
    }
}
