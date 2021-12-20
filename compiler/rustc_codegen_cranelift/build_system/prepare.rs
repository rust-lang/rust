use std::env;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fs;
use std::path::Path;
use std::process::Command;

use crate::rustc_info::{get_file_name, get_rustc_path, get_rustc_version};
use crate::utils::{copy_dir_recursively, spawn_and_wait};

pub(crate) fn prepare() {
    prepare_sysroot();

    eprintln!("[INSTALL] hyperfine");
    Command::new("cargo").arg("install").arg("hyperfine").spawn().unwrap().wait().unwrap();

    clone_repo(
        "rand",
        "https://github.com/rust-random/rand.git",
        "0f933f9c7176e53b2a3c7952ded484e1783f0bf1",
    );
    apply_patches("rand", Path::new("rand"));

    clone_repo(
        "regex",
        "https://github.com/rust-lang/regex.git",
        "341f207c1071f7290e3f228c710817c280c8dca1",
    );

    clone_repo(
        "portable-simd",
        "https://github.com/rust-lang/portable-simd",
        "b8d6b6844602f80af79cd96401339ec594d472d8",
    );
    apply_patches("portable-simd", Path::new("portable-simd"));

    clone_repo(
        "simple-raytracer",
        "https://github.com/ebobby/simple-raytracer",
        "804a7a21b9e673a482797aa289a18ed480e4d813",
    );

    eprintln!("[LLVM BUILD] simple-raytracer");
    let mut build_cmd = Command::new("cargo");
    build_cmd.arg("build").env_remove("CARGO_TARGET_DIR").current_dir("simple-raytracer");
    spawn_and_wait(build_cmd);
    fs::copy(
        Path::new("simple-raytracer/target/debug").join(get_file_name("main", "bin")),
        // FIXME use get_file_name here too once testing is migrated to rust
        "simple-raytracer/raytracer_cg_llvm",
    )
    .unwrap();
}

fn prepare_sysroot() {
    let rustc_path = get_rustc_path();
    let sysroot_src_orig = rustc_path.parent().unwrap().join("../lib/rustlib/src/rust");
    let sysroot_src = env::current_dir().unwrap().join("build_sysroot").join("sysroot_src");

    assert!(sysroot_src_orig.exists());

    if sysroot_src.exists() {
        fs::remove_dir_all(&sysroot_src).unwrap();
    }
    fs::create_dir_all(sysroot_src.join("library")).unwrap();
    eprintln!("[COPY] sysroot src");
    copy_dir_recursively(&sysroot_src_orig.join("library"), &sysroot_src.join("library"));

    let rustc_version = get_rustc_version();
    fs::write(Path::new("build_sysroot").join("rustc_version"), &rustc_version).unwrap();

    eprintln!("[GIT] init");
    let mut git_init_cmd = Command::new("git");
    git_init_cmd.arg("init").arg("-q").current_dir(&sysroot_src);
    spawn_and_wait(git_init_cmd);

    let mut git_add_cmd = Command::new("git");
    git_add_cmd.arg("add").arg(".").current_dir(&sysroot_src);
    spawn_and_wait(git_add_cmd);

    let mut git_commit_cmd = Command::new("git");
    git_commit_cmd
        .arg("commit")
        .arg("-m")
        .arg("Initial commit")
        .arg("-q")
        .current_dir(&sysroot_src);
    spawn_and_wait(git_commit_cmd);

    apply_patches("sysroot", &sysroot_src);

    clone_repo(
        "build_sysroot/compiler-builtins",
        "https://github.com/rust-lang/compiler-builtins.git",
        "0.1.66",
    );
    apply_patches("compiler-builtins", Path::new("build_sysroot/compiler-builtins"));
}

fn clone_repo(target_dir: &str, repo: &str, rev: &str) {
    eprintln!("[CLONE] {}", repo);
    // Ignore exit code as the repo may already have been checked out
    Command::new("git").arg("clone").arg(repo).arg(target_dir).spawn().unwrap().wait().unwrap();

    let mut clean_cmd = Command::new("git");
    clean_cmd.arg("checkout").arg("--").arg(".").current_dir(target_dir);
    spawn_and_wait(clean_cmd);

    let mut checkout_cmd = Command::new("git");
    checkout_cmd.arg("checkout").arg("-q").arg(rev).current_dir(target_dir);
    spawn_and_wait(checkout_cmd);
}

fn get_patches(crate_name: &str) -> Vec<OsString> {
    let mut patches: Vec<_> = fs::read_dir("patches")
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension() == Some(OsStr::new("patch")))
        .map(|path| path.file_name().unwrap().to_owned())
        .filter(|file_name| {
            file_name.to_str().unwrap().split_once("-").unwrap().1.starts_with(crate_name)
        })
        .collect();
    patches.sort();
    patches
}

fn apply_patches(crate_name: &str, target_dir: &Path) {
    for patch in get_patches(crate_name) {
        eprintln!("[PATCH] {:?} <- {:?}", target_dir.file_name().unwrap(), patch);
        let patch_arg = env::current_dir().unwrap().join("patches").join(patch);
        let mut apply_patch_cmd = Command::new("git");
        apply_patch_cmd.arg("am").arg(patch_arg).arg("-q").current_dir(target_dir);
        spawn_and_wait(apply_patch_cmd);
    }
}
