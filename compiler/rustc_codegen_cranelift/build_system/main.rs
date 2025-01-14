#![warn(rust_2018_idioms)]
#![warn(unused_lifetimes)]
#![warn(unreachable_pub)]

use std::path::PathBuf;
use std::{env, process};

use self::utils::Compiler;

mod abi_cafe;
mod bench;
mod build_backend;
mod build_sysroot;
mod config;
mod path;
mod prepare;
mod rustc_info;
mod shared_utils;
mod tests;
mod utils;

fn usage() {
    eprintln!("{}", include_str!("usage.txt"));
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
    AbiCafe,
    Bench,
}

#[derive(Copy, Clone, Debug)]
enum SysrootKind {
    None,
    Clif,
    Llvm,
}

#[derive(Clone, Debug)]
enum CodegenBackend {
    Local(PathBuf),
    Builtin(String),
}

fn main() {
    if env::var_os("RUST_BACKTRACE").is_none() {
        env::set_var("RUST_BACKTRACE", "1");
    }
    env::set_var("CG_CLIF_DISABLE_INCR_CACHE", "1");

    // Force incr comp even in release mode unless in CI or incremental builds are explicitly disabled
    if env::var_os("CARGO_BUILD_INCREMENTAL").is_none() {
        env::set_var("CARGO_BUILD_INCREMENTAL", "true");
    }

    let mut args = env::args().skip(1);
    let command = match args.next().as_deref() {
        Some("prepare") => Command::Prepare,
        Some("build") => Command::Build,
        Some("test") => Command::Test,
        Some("abi-cafe") => Command::AbiCafe,
        Some("bench") => Command::Bench,
        Some(flag) if flag.starts_with('-') => arg_error!("Expected command found flag {}", flag),
        Some(command) => arg_error!("Unknown command {}", command),
        None => {
            usage();
            process::exit(0);
        }
    };

    let mut out_dir = PathBuf::from(".");
    let mut download_dir = None;
    let mut sysroot_kind = SysrootKind::Clif;
    let mut use_unstable_features = true;
    let mut frozen = false;
    let mut skip_tests = vec![];
    let mut use_backend = None;
    while let Some(arg) = args.next().as_deref() {
        match arg {
            "--out-dir" => {
                out_dir = PathBuf::from(args.next().unwrap_or_else(|| {
                    arg_error!("--out-dir requires argument");
                }));
            }
            "--download-dir" => {
                download_dir = Some(PathBuf::from(args.next().unwrap_or_else(|| {
                    arg_error!("--download-dir requires argument");
                })));
            }
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
            "--frozen" => frozen = true,
            "--skip-test" => {
                // FIXME check that all passed in tests actually exist
                skip_tests.push(args.next().unwrap_or_else(|| {
                    arg_error!("--skip-test requires argument");
                }));
            }
            "--use-backend" => {
                use_backend = Some(match args.next() {
                    Some(name) => name,
                    None => arg_error!("--use-backend requires argument"),
                });
            }
            flag if flag.starts_with("-") => arg_error!("Unknown flag {}", flag),
            arg => arg_error!("Unexpected argument {}", arg),
        }
    }

    let current_dir = std::env::current_dir().unwrap();
    out_dir = current_dir.join(out_dir);

    if command == Command::Prepare {
        prepare::prepare(&path::Dirs {
            source_dir: current_dir.clone(),
            download_dir: download_dir
                .map(|dir| current_dir.join(dir))
                .unwrap_or_else(|| out_dir.join("download")),
            build_dir: PathBuf::from("dummy_do_not_use"),
            dist_dir: PathBuf::from("dummy_do_not_use"),
            frozen,
        });
        process::exit(0);
    }

    let rustup_toolchain_name = match (env::var("CARGO"), env::var("RUSTC"), env::var("RUSTDOC")) {
        (Ok(_), Ok(_), Ok(_)) => None,
        (_, Err(_), Err(_)) => Some(rustc_info::get_toolchain_name()),
        vars => {
            eprintln!(
                "If RUSTC or RUSTDOC is set, both need to be set and in addition CARGO needs to be set: {vars:?}"
            );
            process::exit(1);
        }
    };
    let bootstrap_host_compiler = {
        let cargo = rustc_info::get_cargo_path();
        let rustc = rustc_info::get_rustc_path();
        let rustdoc = rustc_info::get_rustdoc_path();
        let triple =
            std::env::var("HOST_TRIPLE").unwrap_or_else(|_| rustc_info::get_host_triple(&rustc));
        Compiler {
            cargo,
            rustc,
            rustdoc,
            rustflags: vec![],
            rustdocflags: vec![],
            triple,
            runner: vec![],
        }
    };
    let target_triple =
        std::env::var("TARGET_TRIPLE").unwrap_or_else(|_| bootstrap_host_compiler.triple.clone());

    let dirs = path::Dirs {
        source_dir: current_dir.clone(),
        download_dir: download_dir
            .map(|dir| current_dir.join(dir))
            .unwrap_or_else(|| out_dir.join("download")),
        build_dir: out_dir.join("build"),
        dist_dir: out_dir.join("dist"),
        frozen,
    };

    std::fs::create_dir_all(&dirs.build_dir).unwrap();

    {
        // Make sure we always explicitly specify the target dir
        let target = dirs.build_dir.join("target_dir_should_be_set_explicitly");
        env::set_var("CARGO_TARGET_DIR", &target);
        let _ = std::fs::remove_file(&target);
        std::fs::File::create(target).unwrap();
    }

    env::set_var("RUSTC", "rustc_should_be_set_explicitly");
    env::set_var("RUSTDOC", "rustdoc_should_be_set_explicitly");

    let cg_clif_dylib = if let Some(name) = use_backend {
        CodegenBackend::Builtin(name)
    } else {
        CodegenBackend::Local(build_backend::build_backend(
            &dirs,
            &bootstrap_host_compiler,
            use_unstable_features,
        ))
    };
    match command {
        Command::Prepare => {
            // Handled above
        }
        Command::Test => {
            tests::run_tests(
                &dirs,
                sysroot_kind,
                use_unstable_features,
                &skip_tests.iter().map(|test| &**test).collect::<Vec<_>>(),
                &cg_clif_dylib,
                &bootstrap_host_compiler,
                rustup_toolchain_name.as_deref(),
                target_triple.clone(),
            );
        }
        Command::AbiCafe => {
            if bootstrap_host_compiler.triple != target_triple {
                eprintln!("Abi-cafe doesn't support cross-compilation");
                process::exit(1);
            }
            abi_cafe::run(
                sysroot_kind,
                &dirs,
                &cg_clif_dylib,
                rustup_toolchain_name.as_deref(),
                &bootstrap_host_compiler,
            );
        }
        Command::Build => {
            build_sysroot::build_sysroot(
                &dirs,
                sysroot_kind,
                &cg_clif_dylib,
                &bootstrap_host_compiler,
                rustup_toolchain_name.as_deref(),
                target_triple,
            );
        }
        Command::Bench => {
            let compiler = build_sysroot::build_sysroot(
                &dirs,
                sysroot_kind,
                &cg_clif_dylib,
                &bootstrap_host_compiler,
                rustup_toolchain_name.as_deref(),
                target_triple,
            );
            bench::benchmark(&dirs, &compiler);
        }
    }
}
