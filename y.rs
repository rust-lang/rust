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
//! $ rustc y.rs -o build/y.bin
//! $ build/y.bin
//! ```
//!
//! # Naming
//!
//! The name `y.rs` was chosen to not conflict with rustc's `x.py`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

#[path = "build_system/build_backend.rs"]
mod build_backend;
#[path = "build_system/build_sysroot.rs"]
mod build_sysroot;
#[path = "build_system/rustc_info.rs"]
mod rustc_info;

fn usage() {
    eprintln!("Usage:");
    eprintln!("  ./y.rs build [--debug] [--sysroot none|clif|llvm] [--target-dir DIR]");
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

    let mut args = env::args().skip(1);
    let command = match args.next().as_deref() {
        Some("prepare") => {
            if args.next().is_some() {
                arg_error!("./x.rs prepare doesn't expect arguments");
            }
            todo!();
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
            flag if flag.starts_with("-") => arg_error!("Unknown flag {}", flag),
            arg => arg_error!("Unexpected argument {}", arg),
        }
    }

    let host_triple = if let Ok(host_triple) = std::env::var("HOST_TRIPLE") {
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
    } else {
        host_triple.clone()
    };

    let cg_clif_dylib = build_backend::build_backend(channel);
    build_sysroot::build_sysroot(
        channel,
        sysroot_kind,
        &target_dir,
        cg_clif_dylib,
        &host_triple,
        &target_triple,
    );
}

#[track_caller]
fn try_hard_link(src: impl AsRef<Path>, dst: impl AsRef<Path>) {
    let src = src.as_ref();
    let dst = dst.as_ref();
    if let Err(_) = fs::hard_link(src, dst) {
        fs::copy(src, dst).unwrap(); // Fallback to copying if hardlinking failed
    }
}
