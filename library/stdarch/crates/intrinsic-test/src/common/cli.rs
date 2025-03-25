use std::path::PathBuf;

/// Intrinsic test tool
#[derive(clap::Parser)]
#[command(
    name = "Intrinsic test tool",
    about = "Generates Rust and C programs for intrinsics and compares the output"
)]
pub struct Cli {
    /// The input file containing the intrinsics
    pub input: PathBuf,

    /// The rust toolchain to use for building the rust code
    #[arg(long)]
    pub toolchain: Option<String>,

    /// The C++ compiler to use for compiling the c++ code
    #[arg(long, default_value_t = String::from("clang++"))]
    pub cppcompiler: String,

    /// Run the C programs under emulation with this command
    #[arg(long)]
    pub runner: Option<String>,

    /// Filename for a list of intrinsics to skip (one per line)
    #[arg(long)]
    pub skip: Option<PathBuf>,

    /// Regenerate test programs, but don't build or run them
    #[arg(long)]
    pub generate_only: bool,

    /// Pass a target the test suite
    #[arg(long, default_value_t = String::from("aarch64-unknown-linux-gnu"))]
    pub target: String,

    /// Set the linker
    #[arg(long)]
    pub linker: Option<String>,

    /// Set the sysroot for the C++ compiler
    #[arg(long)]
    pub cxx_toolchain_dir: Option<String>,
}
