use std::{env, process};

mod build;
mod clean;
mod clone_gcc;
mod config;
mod fmt;
mod info;
mod prepare;
mod rust_tools;
mod rustc_info;
mod test;
mod utils;

const BUILD_DIR: &str = "build";

macro_rules! arg_error {
    ($($err:tt)*) => {{
        eprintln!($($err)*);
        eprintln!();
        usage();
        std::process::exit(1);
    }};
}

fn usage() {
    println!(
        "\
rustc_codegen_gcc build system

Usage: build_system [command] [options]

Options:
        --help    : Displays this help message.

Commands:
        cargo     : Executes a cargo command.
        rustc     : Compiles the program using the GCC compiler.
        clean     : Cleans the build directory, removing all compiled files and artifacts.
        prepare   : Prepares the environment for building, including fetching dependencies and setting up configurations.
        build     : Compiles the project.
        test      : Runs tests for the project.
        info      : Displays information about the build environment and project configuration.
        clone-gcc : Clones the GCC compiler from a specified source.
        fmt       : Runs rustfmt"
    );
}

pub enum Command {
    Cargo,
    Clean,
    CloneGcc,
    Prepare,
    Build,
    Rustc,
    Test,
    Info,
    Fmt,
}

fn main() {
    if env::var("RUST_BACKTRACE").is_err() {
        unsafe {
            env::set_var("RUST_BACKTRACE", "1");
        }
    }

    let command = match env::args().nth(1).as_deref() {
        Some("cargo") => Command::Cargo,
        Some("rustc") => Command::Rustc,
        Some("clean") => Command::Clean,
        Some("prepare") => Command::Prepare,
        Some("build") => Command::Build,
        Some("test") => Command::Test,
        Some("info") => Command::Info,
        Some("clone-gcc") => Command::CloneGcc,
        Some("fmt") => Command::Fmt,
        Some("--help") => {
            usage();
            process::exit(0);
        }
        Some(flag) if flag.starts_with('-') => arg_error!("Expected command found flag {}", flag),
        Some(command) => arg_error!("Unknown command {}", command),
        None => {
            usage();
            process::exit(0);
        }
    };

    if let Err(e) = match command {
        Command::Cargo => rust_tools::run_cargo(),
        Command::Rustc => rust_tools::run_rustc(),
        Command::Clean => clean::run(),
        Command::Prepare => prepare::run(),
        Command::Build => build::run(),
        Command::Test => test::run(),
        Command::Info => info::run(),
        Command::CloneGcc => clone_gcc::run(),
        Command::Fmt => fmt::run(),
    } {
        eprintln!("Command failed to run: {e}");
        process::exit(1);
    }
}
