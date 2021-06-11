use std::ffi::OsStr;
use std::ffi::OsString;
use std::fs;
use std::process::Command;

use crate::utils::spawn_and_wait;

pub(crate) fn prepare() {
    // FIXME implement in rust
    let prepare_sysroot_cmd = Command::new("./build_sysroot/prepare_sysroot_src.sh");
    spawn_and_wait(prepare_sysroot_cmd);

    eprintln!("[INSTALL] hyperfine");
    Command::new("cargo").arg("install").arg("hyperfine").spawn().unwrap().wait().unwrap();

    clone_repo(
        "rand",
        "https://github.com/rust-random/rand.git",
        "0f933f9c7176e53b2a3c7952ded484e1783f0bf1",
    );

    eprintln!("[PATCH] rand");
    for patch in get_patches("crate_patches", "rand") {
        let mut patch_arg = OsString::from("../crate_patches/");
        patch_arg.push(patch);
        let mut apply_patch_cmd = Command::new("git");
        apply_patch_cmd.arg("am").arg(patch_arg).current_dir("rand");
        spawn_and_wait(apply_patch_cmd);
    }

    clone_repo(
        "regex",
        "https://github.com/rust-lang/regex.git",
        "341f207c1071f7290e3f228c710817c280c8dca1",
    );

    clone_repo(
        "simple-raytracer",
        "https://github.com/ebobby/simple-raytracer",
        "804a7a21b9e673a482797aa289a18ed480e4d813",
    );

    eprintln!("[LLVM BUILD] simple-raytracer");
    let mut build_cmd = Command::new("cargo");
    build_cmd.arg("build").env_remove("CARGO_TARGET_DIR").current_dir("simple-raytracer");
    spawn_and_wait(build_cmd);
    fs::copy("simple-raytracer/target/debug/main", "simple-raytracer/raytracer_cg_llvm").unwrap();
}

fn clone_repo(name: &str, repo: &str, commit: &str) {
    eprintln!("[CLONE] {}", repo);
    // Ignore exit code as the repo may already have been checked out
    Command::new("git").arg("clone").arg(repo).spawn().unwrap().wait().unwrap();

    let mut clean_cmd = Command::new("git");
    clean_cmd.arg("checkout").arg("--").arg(".").current_dir(name);
    spawn_and_wait(clean_cmd);

    let mut checkout_cmd = Command::new("git");
    checkout_cmd.arg("checkout").arg(commit).current_dir(name);
    spawn_and_wait(checkout_cmd);
}

fn get_patches(patch_dir: &str, crate_name: &str) -> Vec<OsString> {
    let mut patches: Vec<_> = fs::read_dir(patch_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension() == Some(OsStr::new("patch")))
        .map(|path| path.file_name().unwrap().to_owned())
        .filter(|file_name| file_name.to_str().unwrap().split("-").nth(1).unwrap() == crate_name)
        .collect();
    patches.sort();
    patches
}
