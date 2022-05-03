#![allow(dead_code)]
use super::utils;
use crate::Cli;
use std::{fs::File, path::PathBuf};

pub const ENZYME_VER: &str = "0.0.27";
pub const RUSTC_VER: &str = "1.59.0";

pub fn get_rust_compilation_checkfile_path(args: &Cli) -> PathBuf {
    utils::get_local_rust_repo_path(args.rust.clone()).join("finished-building.txt")
}
pub fn get_enzyme_compilation_checkfile_path(args: &Cli) -> PathBuf {
    utils::get_local_enzyme_repo_path(args.enzyme.clone()).join("finished-building.txt")
}

pub fn is_compiled_rust(args: &Cli) -> bool {
    get_rust_compilation_checkfile_path(args).is_file()
}
pub fn is_compiled_enzyme(args: &Cli) -> bool {
    get_enzyme_compilation_checkfile_path(args).is_file()
}
pub fn set_compiled_rust(args: &Cli) -> Result<(), String> {
    let repo = get_rust_compilation_checkfile_path(args);
    match File::create(repo) {
        Ok(_) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}
pub fn set_compiled_enzyme(args: &Cli) -> Result<(), String> {
    let checkfile = get_enzyme_compilation_checkfile_path(args);
    let rust_repo = utils::get_local_rust_repo_path(args.rust.clone());
    match std::fs::write(
        checkfile,
        "Compiled using: ".to_owned() + &rust_repo.to_string_lossy(),
    ) {
        Ok(_) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}
