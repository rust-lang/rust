use crate::code::version_manager::{
    get_enzyme_compilation_checkfile_path, get_rust_compilation_checkfile_path,
};

use super::utils;
use super::utils::{run_and_printerror, Cli};
use super::version_manager;
use std::{path::PathBuf, process::Command};

#[allow(unused)]
fn print_llvm_version() {
    let mut cmake = Command::new("cmake");
    cmake.args(&["-version"]);
    run_and_printerror(&mut cmake);
}

/// A support function to build Enzyme and Rust such that they can be used together.
///
/// The Rust repository must be build first.
/// It will build a nightly version of Rust, together with LLVM and Clang.
/// Building the Enzyme repository will always run tests to verify that it is working correctly.
pub fn build(args: Cli) -> Result<(), String> {
    let rust_repo = utils::get_local_rust_repo_path(args.rust.clone()); // Default
    if !version_manager::is_compiled_rust(&args) {
        build_rustc(rust_repo.clone());
        version_manager::set_compiled_rust(&args)?;
    } else {
        println!(
            "Skipped compiling rust since the checkfile {} exists",
            get_rust_compilation_checkfile_path(&args).to_string_lossy()
        );
    }

    if !version_manager::is_compiled_enzyme(&args) {
        let enzyme_repo = utils::get_local_enzyme_repo_path(args.enzyme.clone());
        build_enzyme(enzyme_repo, rust_repo);
        version_manager::set_compiled_enzyme(&args)?;
    } else {
        println!(
            "Skipped compiling enzyme since the checkfile {} exists",
            get_enzyme_compilation_checkfile_path(&args).to_string_lossy()
        );
    }

    Ok(())
}

fn build_enzyme(enzyme_repo: PathBuf, rust_repo: PathBuf) {
    let build_path = utils::get_enzyme_build_path(enzyme_repo);
    let llvm_dir = utils::get_llvm_build_path(rust_repo.clone())
        .join("lib")
        .join("cmake")
        .join("llvm");
    let llvm_dir = "-DLLVM_DIR=".to_owned() + llvm_dir.to_str().unwrap();
    let llvm_external_lit = rust_repo
        .join("src")
        .join("llvm-project")
        .join("llvm")
        .join("utils")
        .join("lit")
        .join("lit.py");
    let llvm_external_lit = "-DLLVM_EXTERNAL_LIT=".to_owned() + llvm_external_lit.to_str().unwrap();
    let llvm_external_lib = "-DENZYME_EXTERNAL_SHARED_LIB=ON".to_owned();
    let build_type = "-DCMAKE_BUILD_TYPE=Release";
    let mut cmake = Command::new("cmake");
    let mut ninja = Command::new("ninja");
    let mut ninja_check = Command::new("ninja");
    if !std::path::Path::new(&build_path).exists() {
        std::fs::create_dir(&build_path).unwrap();
    }
    cmake
        .args(&[
            "-G",
            "Ninja",
            "..",
            build_type,
            &llvm_external_lib,
            &llvm_dir,
            &llvm_external_lit,
        ])
        .current_dir(&build_path.to_str().unwrap());
    ninja.current_dir(&build_path.to_str().unwrap());
    ninja_check
        .args(&["check-enzyme"])
        .current_dir(&build_path.to_str().unwrap());
    run_and_printerror(&mut cmake);
    run_and_printerror(&mut ninja);
    run_and_printerror(&mut ninja_check);
}

fn build_rustc(repo_path: PathBuf) {
    let mut cargo = Command::new("cargo");
    let mut configure = Command::new("./configure");
    let mut x = Command::new("x");
    let mut rustup = Command::new("rustup");

    let build_path = utils::get_rustc_build_path(repo_path.clone());
    if !std::path::Path::new(&build_path).exists() {
        std::fs::create_dir(&build_path).unwrap();
    }
    let x_path = std::path::Path::new("src").join("tools").join("x");

    let toolchain_path = utils::get_rustc_stage2_path(repo_path.clone());

    cargo
        .args(&["install", "--path", x_path.to_str().unwrap()])
        .current_dir(&repo_path.to_str().unwrap());

    configure
        .args(&[
            "--enable-llvm-link-shared",
            "--enable-llvm-plugins",
            "--release-channel=nightly", // TODO: Do we still need this? how about nightly?
            "--enable-llvm-assertions",
            "--enable-clang",
            "--enable-lld",
            "--enable-option-checking",
            "--enable-ninja",
        ])
        .current_dir(&repo_path.to_str().unwrap());

    x.args(&["build", "--stage", "2"])
        .current_dir(&build_path.to_str().unwrap());

    rustup
        .args(&[
            "toolchain",
            "link",
            "enzyme",
            toolchain_path.to_str().unwrap(),
        ])
        .current_dir(&build_path.to_str().unwrap());

    run_and_printerror(&mut cargo);
    run_and_printerror(&mut configure);
    run_and_printerror(&mut x);
    run_and_printerror(&mut rustup);
}
