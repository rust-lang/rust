//! A standalone binary for `proc-macro-srv`.
//! Driver for proc macro server
#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]
#![cfg_attr(not(feature = "sysroot-abi"), allow(unused_crate_dependencies))]
#![allow(clippy::print_stdout, clippy::print_stderr)]

#[cfg(feature = "in-rust-tree")]
extern crate rustc_driver as _;

mod version;

#[cfg(feature = "sysroot-abi")]
mod main_loop;
use clap::{Command, ValueEnum};
#[cfg(feature = "sysroot-abi")]
use main_loop::run;

fn main() -> std::io::Result<()> {
    let v = std::env::var("RUST_ANALYZER_INTERNALS_DO_NOT_USE");
    if v.is_err() {
        eprintln!(
            "This is an IDE implementation detail, you can use this tool by exporting RUST_ANALYZER_INTERNALS_DO_NOT_USE."
        );
        eprintln!(
            "Note that this tool's API is highly unstable and may break without prior notice"
        );
        std::process::exit(122);
    }
    let matches = Command::new("proc-macro-srv")
        .args(&[
            clap::Arg::new("format")
                .long("format")
                .action(clap::ArgAction::Set)
                .default_value("json-legacy")
                .value_parser(clap::builder::EnumValueParser::<ProtocolFormat>::new()),
            clap::Arg::new("version")
                .long("version")
                .action(clap::ArgAction::SetTrue)
                .help("Prints the version of the proc-macro-srv"),
        ])
        .get_matches();
    if matches.get_flag("version") {
        println!("rust-analyzer-proc-macro-srv {}", version::version());
        return Ok(());
    }
    let &format =
        matches.get_one::<ProtocolFormat>("format").expect("format value should always be present");
    run(format)
}

#[derive(Copy, Clone)]
enum ProtocolFormat {
    JsonLegacy,
    PostcardLegacy,
}

impl ValueEnum for ProtocolFormat {
    fn value_variants<'a>() -> &'a [Self] {
        &[ProtocolFormat::JsonLegacy, ProtocolFormat::PostcardLegacy]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        match self {
            ProtocolFormat::JsonLegacy => Some(clap::builder::PossibleValue::new("json-legacy")),
            ProtocolFormat::PostcardLegacy => {
                Some(clap::builder::PossibleValue::new("postcard-legacy"))
            }
        }
    }
    fn from_str(input: &str, _ignore_case: bool) -> Result<Self, String> {
        match input {
            "json-legacy" => Ok(ProtocolFormat::JsonLegacy),
            "postcard-legacy" => Ok(ProtocolFormat::PostcardLegacy),
            _ => Err(format!("unknown protocol format: {input}")),
        }
    }
}

#[cfg(not(feature = "sysroot-abi"))]
fn run(_: ProtocolFormat) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "proc-macro-srv-cli needs to be compiled with the `sysroot-abi` feature to function"
            .to_owned(),
    ))
}
