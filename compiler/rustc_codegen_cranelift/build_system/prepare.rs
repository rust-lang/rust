use std::env;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fs;
use std::path::Path;
use std::process::Command;

use super::rustc_info::{get_file_name, get_rustc_path, get_rustc_version};
use super::utils::{copy_dir_recursively, spawn_and_wait};

pub(crate) fn prepare() {
    prepare_sysroot();

    eprintln!("[INSTALL] hyperfine");
    Command::new("cargo").arg("install").arg("hyperfine").spawn().unwrap().wait().unwrap();

    clone_repo_shallow_github(
        "abi-checker",
        "Gankra",
        "abi-checker",
        "a2232d45f202846f5c02203c9f27355360f9a2ff",
    );
    apply_patches("abi-checker", Path::new("abi-checker"));

    clone_repo_shallow_github(
        "rand",
        "rust-random",
        "rand",
        "0f933f9c7176e53b2a3c7952ded484e1783f0bf1",
    );
    apply_patches("rand", Path::new("rand"));

    clone_repo_shallow_github(
        "regex",
        "rust-lang",
        "regex",
        "341f207c1071f7290e3f228c710817c280c8dca1",
    );

    clone_repo_shallow_github(
        "portable-simd",
        "rust-lang",
        "portable-simd",
        "b8d6b6844602f80af79cd96401339ec594d472d8",
    );
    apply_patches("portable-simd", Path::new("portable-simd"));

    clone_repo_shallow_github(
        "simple-raytracer",
        "ebobby",
        "simple-raytracer",
        "804a7a21b9e673a482797aa289a18ed480e4d813",
    );

    eprintln!("[LLVM BUILD] simple-raytracer");
    let mut build_cmd = Command::new("cargo");
    build_cmd.arg("build").env_remove("CARGO_TARGET_DIR").current_dir("simple-raytracer");
    spawn_and_wait(build_cmd);
    fs::copy(
        Path::new("simple-raytracer/target/debug").join(get_file_name("main", "bin")),
        Path::new("simple-raytracer").join(get_file_name("raytracer_cg_llvm", "bin")),
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

    init_git_repo(&sysroot_src);

    apply_patches("sysroot", &sysroot_src);
}

#[allow(dead_code)]
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

fn clone_repo_shallow_github(target_dir: &str, username: &str, repo: &str, rev: &str) {
    if cfg!(windows) {
        // Older windows doesn't have tar or curl by default. Fall back to using git.
        clone_repo(target_dir, &format!("https://github.com/{}/{}.git", username, repo), rev);
        return;
    }

    let archive_url = format!("https://github.com/{}/{}/archive/{}.tar.gz", username, repo, rev);
    let archive_file = format!("{}.tar.gz", rev);
    let archive_dir = format!("{}-{}", repo, rev);

    eprintln!("[DOWNLOAD] {}/{} from {}", username, repo, archive_url);

    // Remove previous results if they exists
    let _ = std::fs::remove_file(&archive_file);
    let _ = std::fs::remove_dir_all(&archive_dir);
    let _ = std::fs::remove_dir_all(target_dir);

    // Download zip archive
    let mut download_cmd = Command::new("curl");
    download_cmd.arg("--location").arg("--output").arg(&archive_file).arg(archive_url);
    spawn_and_wait(download_cmd);

    // Unpack tar archive
    let mut unpack_cmd = Command::new("tar");
    unpack_cmd.arg("xf").arg(&archive_file);
    spawn_and_wait(unpack_cmd);

    // Rename unpacked dir to the expected name
    std::fs::rename(archive_dir, target_dir).unwrap();

    init_git_repo(Path::new(target_dir));

    // Cleanup
    std::fs::remove_file(archive_file).unwrap();
}

fn init_git_repo(repo_dir: &Path) {
    let mut git_init_cmd = Command::new("git");
    git_init_cmd.arg("init").arg("-q").current_dir(repo_dir);
    spawn_and_wait(git_init_cmd);

    let mut git_add_cmd = Command::new("git");
    git_add_cmd.arg("add").arg(".").current_dir(repo_dir);
    spawn_and_wait(git_add_cmd);

    let mut git_commit_cmd = Command::new("git");
    git_commit_cmd.arg("commit").arg("-m").arg("Initial commit").arg("-q").current_dir(repo_dir);
    spawn_and_wait(git_commit_cmd);
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
