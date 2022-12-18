use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::build_sysroot::{SYSROOT_RUSTC_VERSION, SYSROOT_SRC};
use super::path::{Dirs, RelPath};
use super::rustc_info::{get_file_name, get_rustc_path, get_rustc_version};
use super::utils::{copy_dir_recursively, spawn_and_wait, Compiler};

pub(crate) fn prepare(dirs: &Dirs) {
    if RelPath::DOWNLOAD.to_path(dirs).exists() {
        std::fs::remove_dir_all(RelPath::DOWNLOAD.to_path(dirs)).unwrap();
    }
    std::fs::create_dir_all(RelPath::DOWNLOAD.to_path(dirs)).unwrap();

    prepare_sysroot(dirs);

    // FIXME maybe install this only locally?
    eprintln!("[INSTALL] hyperfine");
    Command::new("cargo")
        .arg("install")
        .arg("hyperfine")
        .env_remove("CARGO_TARGET_DIR")
        .spawn()
        .unwrap()
        .wait()
        .unwrap();

    super::abi_cafe::ABI_CAFE_REPO.fetch(dirs);
    super::tests::RAND_REPO.fetch(dirs);
    super::tests::REGEX_REPO.fetch(dirs);
    super::tests::PORTABLE_SIMD_REPO.fetch(dirs);
    super::tests::SIMPLE_RAYTRACER_REPO.fetch(dirs);

    eprintln!("[LLVM BUILD] simple-raytracer");
    let host_compiler = Compiler::host();
    let build_cmd = super::tests::SIMPLE_RAYTRACER.build(&host_compiler, dirs);
    spawn_and_wait(build_cmd);
    fs::copy(
        super::tests::SIMPLE_RAYTRACER
            .target_dir(dirs)
            .join(&host_compiler.triple)
            .join("debug")
            .join(get_file_name("main", "bin")),
        RelPath::BUILD.to_path(dirs).join(get_file_name("raytracer_cg_llvm", "bin")),
    )
    .unwrap();
}

fn prepare_sysroot(dirs: &Dirs) {
    let rustc_path = get_rustc_path();
    let sysroot_src_orig = rustc_path.parent().unwrap().join("../lib/rustlib/src/rust");
    let sysroot_src = SYSROOT_SRC;

    assert!(sysroot_src_orig.exists());

    sysroot_src.ensure_fresh(dirs);
    fs::create_dir_all(sysroot_src.to_path(dirs).join("library")).unwrap();
    eprintln!("[COPY] sysroot src");
    copy_dir_recursively(
        &sysroot_src_orig.join("library"),
        &sysroot_src.to_path(dirs).join("library"),
    );

    let rustc_version = get_rustc_version();
    fs::write(SYSROOT_RUSTC_VERSION.to_path(dirs), &rustc_version).unwrap();

    eprintln!("[GIT] init");
    init_git_repo(&sysroot_src.to_path(dirs));

    apply_patches(dirs, "sysroot", &sysroot_src.to_path(dirs));
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
    pub(crate) const fn github(
        user: &'static str,
        repo: &'static str,
        rev: &'static str,
        patch_name: &'static str,
    ) -> GitRepo {
        GitRepo { url: GitRepoUrl::Github { user, repo }, rev, patch_name }
    }

    pub(crate) const fn source_dir(&self) -> RelPath {
        match self.url {
            GitRepoUrl::Github { user: _, repo } => RelPath::DOWNLOAD.join(repo),
        }
    }

    fn fetch(&self, dirs: &Dirs) {
        match self.url {
            GitRepoUrl::Github { user, repo } => {
                clone_repo_shallow_github(
                    dirs,
                    &self.source_dir().to_path(dirs),
                    user,
                    repo,
                    self.rev,
                );
            }
        }
        apply_patches(dirs, self.patch_name, &self.source_dir().to_path(dirs));
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

fn clone_repo_shallow_github(dirs: &Dirs, download_dir: &Path, user: &str, repo: &str, rev: &str) {
    if cfg!(windows) {
        // Older windows doesn't have tar or curl by default. Fall back to using git.
        clone_repo(download_dir, &format!("https://github.com/{}/{}.git", user, repo), rev);
        return;
    }

    let archive_url = format!("https://github.com/{}/{}/archive/{}.tar.gz", user, repo, rev);
    let archive_file = RelPath::DOWNLOAD.to_path(dirs).join(format!("{}.tar.gz", rev));
    let archive_dir = RelPath::DOWNLOAD.to_path(dirs).join(format!("{}-{}", repo, rev));

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
    unpack_cmd.arg("xf").arg(&archive_file).current_dir(RelPath::DOWNLOAD.to_path(dirs));
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
    git_commit_cmd
        .arg("-c")
        .arg("user.name=Dummy")
        .arg("-c")
        .arg("user.email=dummy@example.com")
        .arg("commit")
        .arg("-m")
        .arg("Initial commit")
        .arg("-q")
        .current_dir(repo_dir);
    spawn_and_wait(git_commit_cmd);
}

fn get_patches(dirs: &Dirs, crate_name: &str) -> Vec<PathBuf> {
    let mut patches: Vec<_> = fs::read_dir(RelPath::PATCHES.to_path(dirs))
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

fn apply_patches(dirs: &Dirs, crate_name: &str, target_dir: &Path) {
    if crate_name == "<none>" {
        return;
    }

    for patch in get_patches(dirs, crate_name) {
        eprintln!(
            "[PATCH] {:?} <- {:?}",
            target_dir.file_name().unwrap(),
            patch.file_name().unwrap()
        );
        let mut apply_patch_cmd = Command::new("git");
        apply_patch_cmd
            .arg("-c")
            .arg("user.name=Dummy")
            .arg("-c")
            .arg("user.email=dummy@example.com")
            .arg("am")
            .arg(patch)
            .arg("-q")
            .current_dir(target_dir);
        spawn_and_wait(apply_patch_cmd);
    }
}
