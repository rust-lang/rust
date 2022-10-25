use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::rustc_info::{get_file_name, get_rustc_path, get_rustc_version};
use super::utils::{cargo_command, copy_dir_recursively, spawn_and_wait};

pub(crate) const ABI_CAFE: GitRepo =
    GitRepo::github("Gankra", "abi-cafe", "4c6dc8c9c687e2b3a760ff2176ce236872b37212", "abi-cafe");

pub(crate) const RAND: GitRepo =
    GitRepo::github("rust-random", "rand", "0f933f9c7176e53b2a3c7952ded484e1783f0bf1", "rand");

pub(crate) const REGEX: GitRepo =
    GitRepo::github("rust-lang", "regex", "341f207c1071f7290e3f228c710817c280c8dca1", "regex");

pub(crate) const PORTABLE_SIMD: GitRepo = GitRepo::github(
    "rust-lang",
    "portable-simd",
    "d5cd4a8112d958bd3a252327e0d069a6363249bd",
    "portable-simd",
);

pub(crate) const SIMPLE_RAYTRACER: GitRepo = GitRepo::github(
    "ebobby",
    "simple-raytracer",
    "804a7a21b9e673a482797aa289a18ed480e4d813",
    "<none>",
);

pub(crate) fn prepare() {
    if Path::new("download").exists() {
        std::fs::remove_dir_all(Path::new("download")).unwrap();
    }
    std::fs::create_dir_all(Path::new("download")).unwrap();

    prepare_sysroot();

    // FIXME maybe install this only locally?
    eprintln!("[INSTALL] hyperfine");
    Command::new("cargo").arg("install").arg("hyperfine").spawn().unwrap().wait().unwrap();

    ABI_CAFE.fetch();
    RAND.fetch();
    REGEX.fetch();
    PORTABLE_SIMD.fetch();
    SIMPLE_RAYTRACER.fetch();

    eprintln!("[LLVM BUILD] simple-raytracer");
    let build_cmd = cargo_command("cargo", "build", None, &SIMPLE_RAYTRACER.source_dir());
    spawn_and_wait(build_cmd);
    fs::copy(
        SIMPLE_RAYTRACER
            .source_dir()
            .join("target")
            .join("debug")
            .join(get_file_name("main", "bin")),
        SIMPLE_RAYTRACER.source_dir().join(get_file_name("raytracer_cg_llvm", "bin")),
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

pub(crate) struct GitRepo {
    url: GitRepoUrl,
    rev: &'static str,
    patch_name: &'static str,
}

enum GitRepoUrl {
    Github { user: &'static str, repo: &'static str },
}

impl GitRepo {
    const fn github(
        user: &'static str,
        repo: &'static str,
        rev: &'static str,
        patch_name: &'static str,
    ) -> GitRepo {
        GitRepo { url: GitRepoUrl::Github { user, repo }, rev, patch_name }
    }

    pub(crate) fn source_dir(&self) -> PathBuf {
        match self.url {
            GitRepoUrl::Github { user: _, repo } => {
                std::env::current_dir().unwrap().join("download").join(repo)
            }
        }
    }

    fn fetch(&self) {
        match self.url {
            GitRepoUrl::Github { user, repo } => {
                clone_repo_shallow_github(&self.source_dir(), user, repo, self.rev);
            }
        }
        apply_patches(self.patch_name, &self.source_dir());
    }
}

#[allow(dead_code)]
fn clone_repo(download_dir: &Path, repo: &str, rev: &str) {
    eprintln!("[CLONE] {}", repo);
    // Ignore exit code as the repo may already have been checked out
    Command::new("git").arg("clone").arg(repo).arg(&download_dir).spawn().unwrap().wait().unwrap();

    let mut clean_cmd = Command::new("git");
    clean_cmd.arg("checkout").arg("--").arg(".").current_dir(&download_dir);
    spawn_and_wait(clean_cmd);

    let mut checkout_cmd = Command::new("git");
    checkout_cmd.arg("checkout").arg("-q").arg(rev).current_dir(download_dir);
    spawn_and_wait(checkout_cmd);
}

fn clone_repo_shallow_github(download_dir: &Path, user: &str, repo: &str, rev: &str) {
    if cfg!(windows) {
        // Older windows doesn't have tar or curl by default. Fall back to using git.
        clone_repo(download_dir, &format!("https://github.com/{}/{}.git", user, repo), rev);
        return;
    }

    let downloads_dir = std::env::current_dir().unwrap().join("download");

    let archive_url = format!("https://github.com/{}/{}/archive/{}.tar.gz", user, repo, rev);
    let archive_file = downloads_dir.join(format!("{}.tar.gz", rev));
    let archive_dir = downloads_dir.join(format!("{}-{}", repo, rev));

    eprintln!("[DOWNLOAD] {}/{} from {}", user, repo, archive_url);

    // Remove previous results if they exists
    let _ = std::fs::remove_file(&archive_file);
    let _ = std::fs::remove_dir_all(&archive_dir);
    let _ = std::fs::remove_dir_all(&download_dir);

    // Download zip archive
    let mut download_cmd = Command::new("curl");
    download_cmd.arg("--location").arg("--output").arg(&archive_file).arg(archive_url);
    spawn_and_wait(download_cmd);

    // Unpack tar archive
    let mut unpack_cmd = Command::new("tar");
    unpack_cmd.arg("xf").arg(&archive_file).current_dir(downloads_dir);
    spawn_and_wait(unpack_cmd);

    // Rename unpacked dir to the expected name
    std::fs::rename(archive_dir, &download_dir).unwrap();

    init_git_repo(&download_dir);

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

fn get_patches(source_dir: &Path, crate_name: &str) -> Vec<PathBuf> {
    let mut patches: Vec<_> = fs::read_dir(source_dir.join("patches"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension() == Some(OsStr::new("patch")))
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .split_once("-")
                .unwrap()
                .1
                .starts_with(crate_name)
        })
        .collect();
    patches.sort();
    patches
}

fn apply_patches(crate_name: &str, target_dir: &Path) {
    if crate_name == "<none>" {
        return;
    }

    for patch in get_patches(&std::env::current_dir().unwrap(), crate_name) {
        eprintln!(
            "[PATCH] {:?} <- {:?}",
            target_dir.file_name().unwrap(),
            patch.file_name().unwrap()
        );
        let mut apply_patch_cmd = Command::new("git");
        apply_patch_cmd.arg("am").arg(patch).arg("-q").current_dir(target_dir);
        spawn_and_wait(apply_patch_cmd);
    }
}
