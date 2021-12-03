#!/usr/bin/env bash
#![allow()] /*This line is ignored by bash
# This block is ignored by rustc
set -e
echo "[BUILD] y.rs" 1>&2
rustc $0 -o ${0/.rs/.bin} -g
exec ${0/.rs/.bin} $@
*/

//! The build system for cg_clif
//!
//! # Manual compilation
//!
//! If your system doesn't support shell scripts you can manually compile and run this file using
//! for example:
//!
//! ```shell
//! $ rustc y.rs -o y.bin
//! $ ./y.bin
//! ```
//!
//! # Naming
//!
//! The name `y.rs` was chosen to not conflict with rustc's `x.py`.

use std::env;
use std::path::PathBuf;
use std::process;

#[path = "build_system/build_backend.rs"]
mod build_backend;
#[path = "build_system/build_sysroot.rs"]
mod build_sysroot;
#[path = "build_system/config.rs"]
mod config;
#[path = "build_system/prepare.rs"]
mod prepare;
#[path = "build_system/rustc_info.rs"]
mod rustc_info;
#[path = "build_system/utils.rs"]
mod utils;

fn usage() {
    eprintln!("Usage:");
    eprintln!("  ./y.rs prepare");
    eprintln!(
        "  ./y.rs build [--debug] [--sysroot none|clif|llvm] [--target-dir DIR] [--no-unstable-features]"
    );
}

macro_rules! arg_error {
    ($($err:tt)*) => {{
        eprintln!($($err)*);
        usage();
        std::process::exit(1);
    }};
}

enum Command {
    Build,
}

#[derive(Copy, Clone)]
enum SysrootKind {
    None,
    Clif,
    Llvm,
}

fn main() {
    env::set_var("CG_CLIF_DISPLAY_CG_TIME", "1");
    env::set_var("CG_CLIF_DISABLE_INCR_CACHE", "1");
    // The target dir is expected in the default location. Guard against the user changing it.
    env::set_var("CARGO_TARGET_DIR", "target");

    let mut args = env::args().skip(1);
    let command = match args.next().as_deref() {
        Some("prepare") => {
            if args.next().is_some() {
                arg_error!("./x.rs prepare doesn't expect arguments");
            }
            prepare::prepare();
            process::exit(0);
        }
        Some("build") => Command::Build,
        Some(flag) if flag.starts_with('-') => arg_error!("Expected command found flag {}", flag),
        Some(command) => arg_error!("Unknown command {}", command),
        None => {
            usage();
            process::exit(0);
        }
    };

    let mut target_dir = PathBuf::from("build");
    let mut channel = "release";
    let mut sysroot_kind = SysrootKind::Clif;
    let mut use_unstable_features = true;
    while let Some(arg) = args.next().as_deref() {
        match arg {
            "--target-dir" => {
                target_dir = PathBuf::from(args.next().unwrap_or_else(|| {
                    arg_error!("--target-dir requires argument");
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
    } else if let Some(host_triple) = crate::config::get_value("host") {
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
    } else if let Some(target_triple) = crate::config::get_value("target") {
        target_triple
    } else {
        host_triple.clone()
    };

    if target_triple.ends_with("-msvc") {
        eprintln!("The MSVC toolchain is not yet supported by rustc_codegen_cranelift.");
        eprintln!("Switch to the MinGW toolchain for Windows support.");
        eprintln!("Hint: You can use `rustup set default-host x86_64-pc-windows-gnu` to");
        eprintln!("set the global default target to MinGW");
        process::exit(1);
    }

    let cg_clif_build_dir =
        build_backend::build_backend(channel, &host_triple, use_unstable_features);
    build_sysroot::build_sysroot(
        channel,
        sysroot_kind,
        &target_dir,
        cg_clif_build_dir,
        &host_triple,
        &target_triple,
    );
}
