use std::{env, process};

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

macro_rules! arg_error {
    ($($err:tt)*) => {{
        eprintln!($($err)*);
        eprintln!();
        usage();
        std::process::exit(1);
    }};
}

macro_rules! commands_decl {
    ($($variant:ident: $doc_name:literal => $doc:literal ,)+) => {
        enum Command {
            $($variant),+
        }

        impl<'a> From<Option<&'a str>> for Command {
            fn from(arg: Option<&'a str>) -> Self {
                match arg {
                    $(Some($doc_name) => Self::$variant,)+
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
                }
            }
        }

        fn usage() {
            println!("\
rustc_codegen_gcc build system

Usage: build_system [command] [options]

Options:
    --help     : Displays this help message.

Commands:",
            );
            let mut commands = vec![$(($doc_name, $doc),)+];
            let longest = commands.iter().map(|(name, _)| name.len()).max().unwrap();

            commands.sort_unstable_by(|a, b| a.0.cmp(b.0));
            for (name, doc) in commands {
                let spacing = std::iter::repeat(' ').take(longest - name.len() + 1).collect::<String>();
                eprintln!("    {name}{spacing}: {doc}.");
            }
        }
    }
}

commands_decl! {
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
        process::exit(1);
    }
}
