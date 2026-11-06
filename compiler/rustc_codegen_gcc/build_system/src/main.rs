use std::{env, process};

#[macro_use]
mod macros;
mod abi_test;
mod build;
mod clean;
mod clippy;
mod clone_gcc;
mod config;
mod fmt;
mod fuzz;
mod info;
mod prepare;
mod rust_tools;
mod rustc_info;
mod test;
mod todo;
mod utils;
const BUILD_DIR: &str = "build";

commands_decl! {
    Command,
    Cargo: "cargo" => "Executes a cargo command",
    Clean: "clean" => "Cleans the build directory, removing all compiled files and artifacts",
    Clippy: "clippy" => "Runs clippy",
    CloneGcc: "clone-gcc" => "Clones the GCC compiler from a specified source",
    Prepare: "prepare" => "Prepares the environment for building, including fetching dependencies and setting up configurations",
    Build: "build" => "Compiles the project",
    Rustc: "rustc" => "Compiles the program using the GCC compiler",
    Test: "test" => "Runs tests for the project",
    Info: "info" => "Displays information about the build environment and project configuration",
    Fmt: "fmt" => "Runs rustfmt",
    Fuzz: "fuzz" => "Fuzzes `cg_gcc` using `rustlantis`",
    AbiTest: "abi-test" => "Runs the abi-cafe test suite on the codegen, checking for ABI compatibility with LLVM",
    CheckTodo: "check-todo" => "Checks todo in the project",
}

fn main() {
    if env::var("RUST_BACKTRACE").is_err() {
        unsafe {
            env::set_var("RUST_BACKTRACE", "1");
        }
    }

    if let Err(e) = match Command::from(env::args().nth(1).as_deref()) {
        Command::Cargo => rust_tools::run_cargo(),
        Command::Rustc => rust_tools::run_rustc(),
        Command::Clean => clean::run(),
        Command::Prepare => prepare::run(),
        Command::Build => build::run(),
        Command::Test => test::run(),
        Command::Info => info::run(),
        Command::CloneGcc => clone_gcc::run(),
        Command::Fmt => fmt::run(),
        Command::Fuzz => fuzz::run(),
        Command::AbiTest => abi_test::run(),
        Command::Clippy => clippy::run(),
        Command::CheckTodo => todo::run(),
    } {
        eprintln!("Command failed to run: {e}");
        // CI needs to tell a build system error apart from the test failures some suites expect.
        let exit_code =
            if e == test::TESTS_FAILED_ERROR { test::TESTS_FAILED_EXIT_CODE } else { 1 };
        process::exit(exit_code);
    }
}
