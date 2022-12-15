use std::env;
use std::path::PathBuf;
use std::process;

use self::utils::is_ci;

mod abi_cafe;
mod build_backend;
mod build_sysroot;
mod config;
mod path;
mod prepare;
mod rustc_info;
mod tests;
mod utils;

const USAGE: &str = r#"The build system of cg_clif.

USAGE:
    ./y.rs prepare [--out-dir DIR]
    ./y.rs build [--debug] [--sysroot none|clif|llvm] [--out-dir DIR] [--no-unstable-features]
    ./y.rs test [--debug] [--sysroot none|clif|llvm] [--out-dir DIR] [--no-unstable-features]

OPTIONS:
    --sysroot none|clif|llvm
            Which sysroot libraries to use:
            `none` will not include any standard library in the sysroot.
            `clif` will build the standard library using Cranelift.
            `llvm` will use the pre-compiled standard library of rustc which is compiled with LLVM.

    --out-dir DIR
            Specify the directory in which the download, build and dist directories are stored.
            By default this is the working directory.

    --no-unstable-features
            fSome features are not yet ready for production usage. This option will disable these
            features. This includes the JIT mode and inline assembly support.
"#;

fn usage() {
    eprintln!("{USAGE}");
}

macro_rules! arg_error {
    ($($err:tt)*) => {{
        eprintln!($($err)*);
        usage();
        std::process::exit(1);
    }};
}

#[derive(PartialEq, Debug)]
enum Command {
    Prepare,
    Build,
    Test,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum SysrootKind {
    None,
    Clif,
    Llvm,
}

pub fn main() {
    if env::var("RUST_BACKTRACE").is_err() {
        env::set_var("RUST_BACKTRACE", "1");
    }
    env::set_var("CG_CLIF_DISPLAY_CG_TIME", "1");
    env::set_var("CG_CLIF_DISABLE_INCR_CACHE", "1");

    if is_ci() {
        // Disabling incr comp reduces cache size and incr comp doesn't save as much on CI anyway
        env::set_var("CARGO_BUILD_INCREMENTAL", "false");

        // Enable the Cranelift verifier
        env::set_var("CG_CLIF_ENABLE_VERIFIER", "1");
    }

    let mut args = env::args().skip(1);
    let command = match args.next().as_deref() {
        Some("prepare") => Command::Prepare,
        Some("build") => Command::Build,
        Some("test") => Command::Test,
        Some(flag) if flag.starts_with('-') => arg_error!("Expected command found flag {}", flag),
        Some(command) => arg_error!("Unknown command {}", command),
        None => {
            usage();
            process::exit(0);
        }
    };

    let mut out_dir = PathBuf::from(".");
    let mut channel = "release";
    let mut sysroot_kind = SysrootKind::Clif;
    let mut use_unstable_features = true;
    while let Some(arg) = args.next().as_deref() {
        match arg {
            "--out-dir" => {
                out_dir = PathBuf::from(args.next().unwrap_or_else(|| {
                    arg_error!("--out-dir requires argument");
                }))
            }
            "--debug" => channel = "debug",
            "--sysroot" => {
                sysroot_kind = match args.next().as_deref() {
                    Some("none") => SysrootKind::None,
                    Some("clif") => SysrootKind::Clif,
                    Some("llvm") => SysrootKind::Llvm,
                    Some(arg) => arg_error!("Unknown sysroot kind {}", arg),
                    None => arg_error!("--sysroot requires argument"),
                }
            }
            "--no-unstable-features" => use_unstable_features = false,
            flag if flag.starts_with("-") => arg_error!("Unknown flag {}", flag),
            arg => arg_error!("Unexpected argument {}", arg),
        }
    }

    let host_triple = if let Ok(host_triple) = std::env::var("HOST_TRIPLE") {
        host_triple
    } else if let Some(host_triple) = config::get_value("host") {
        host_triple
    } else {
        rustc_info::get_host_triple()
    };
    let target_triple = if let Ok(target_triple) = std::env::var("TARGET_TRIPLE") {
        if target_triple != "" {
            target_triple
        } else {
            host_triple.clone() // Empty target triple can happen on GHA
        }
    } else if let Some(target_triple) = config::get_value("target") {
        target_triple
    } else {
        host_triple.clone()
    };

    // FIXME allow changing the location of these dirs using cli arguments
    let current_dir = std::env::current_dir().unwrap();
    out_dir = current_dir.join(out_dir);
    let dirs = path::Dirs {
        source_dir: current_dir.clone(),
        download_dir: out_dir.join("download"),
        build_dir: out_dir.join("build"),
        dist_dir: out_dir.join("dist"),
    };

    path::RelPath::BUILD.ensure_exists(&dirs);

    {
        // Make sure we always explicitly specify the target dir
        let target =
            path::RelPath::BUILD.join("target_dir_should_be_set_explicitly").to_path(&dirs);
        env::set_var("CARGO_TARGET_DIR", &target);
        let _ = std::fs::remove_file(&target);
        std::fs::File::create(target).unwrap();
    }

    if command == Command::Prepare {
        prepare::prepare(&dirs);
        process::exit(0);
    }

    let cg_clif_dylib =
        build_backend::build_backend(&dirs, channel, &host_triple, use_unstable_features);
    match command {
        Command::Prepare => {
            // Handled above
        }
        Command::Test => {
            tests::run_tests(
                &dirs,
                channel,
                sysroot_kind,
                &cg_clif_dylib,
                &host_triple,
                &target_triple,
            );

            abi_cafe::run(
                channel,
                sysroot_kind,
                &dirs,
                &cg_clif_dylib,
                &host_triple,
                &target_triple,
            );
        }
        Command::Build => {
            build_sysroot::build_sysroot(
                &dirs,
                channel,
                sysroot_kind,
                &cg_clif_dylib,
                &host_triple,
                &target_triple,
            );
        }
    }
}
