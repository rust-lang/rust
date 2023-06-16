use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::build_sysroot::STDLIB_SRC;
use super::path::{Dirs, RelPath};
use super::rustc_info::get_default_sysroot;
use super::utils::{
    copy_dir_recursively, git_command, remove_dir_if_exists, retry_spawn_and_wait, spawn_and_wait,
};

pub(crate) fn prepare(dirs: &Dirs) {
    RelPath::DOWNLOAD.ensure_exists(dirs);
    super::tests::RAND_REPO.fetch(dirs);
    super::tests::REGEX_REPO.fetch(dirs);
    super::tests::PORTABLE_SIMD_REPO.fetch(dirs);
}

pub(crate) fn prepare_stdlib(dirs: &Dirs, rustc: &Path) {
    let sysroot_src_orig = get_default_sysroot(rustc).join("lib/rustlib/src/rust");
    assert!(sysroot_src_orig.exists());

    apply_patches(dirs, "stdlib", &sysroot_src_orig, &STDLIB_SRC.to_path(dirs));

    std::fs::write(
        STDLIB_SRC.to_path(dirs).join("Cargo.toml"),
        r#"
[workspace]
members = ["./library/sysroot"]

[patch.crates-io]
rustc-std-workspace-core = { path = "./library/rustc-std-workspace-core" }
rustc-std-workspace-alloc = { path = "./library/rustc-std-workspace-alloc" }
rustc-std-workspace-std = { path = "./library/rustc-std-workspace-std" }

# Mandatory for correctly compiling compiler-builtins
[profile.dev.package.compiler_builtins]
debug-assertions = false
overflow-checks = false
codegen-units = 10000

[profile.release.package.compiler_builtins]
debug-assertions = false
overflow-checks = false
codegen-units = 10000
"#,
    )
    .unwrap();

    let source_lockfile = RelPath::PATCHES.to_path(dirs).join("stdlib-lock.toml");
    let target_lockfile = STDLIB_SRC.to_path(dirs).join("Cargo.lock");
    fs::copy(source_lockfile, target_lockfile).unwrap();
}

pub(crate) struct GitRepo {
    url: GitRepoUrl,
    rev: &'static str,
    content_hash: &'static str,
    patch_name: &'static str,
}

enum GitRepoUrl {
    Github { user: &'static str, repo: &'static str },
}

// Note: This uses a hasher which is not cryptographically secure. This is fine as the hash is meant
// to protect against accidental modification and outdated downloads, not against manipulation.
fn hash_file(file: &std::path::Path) -> u64 {
    let contents = std::fs::read(file).unwrap();
    #[allow(deprecated)]
    let mut hasher = std::hash::SipHasher::new();
    std::hash::Hash::hash(&contents, &mut hasher);
    std::hash::Hasher::finish(&hasher)
}

fn hash_dir(dir: &std::path::Path) -> u64 {
    let mut sub_hashes = std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            sub_hashes
                .insert(entry.file_name().to_str().unwrap().to_owned(), hash_dir(&entry.path()));
        } else {
            sub_hashes
                .insert(entry.file_name().to_str().unwrap().to_owned(), hash_file(&entry.path()));
        }
    }
    #[allow(deprecated)]
    let mut hasher = std::hash::SipHasher::new();
    std::hash::Hash::hash(&sub_hashes, &mut hasher);
    std::hash::Hasher::finish(&hasher)
}

impl GitRepo {
    pub(crate) const fn github(
        user: &'static str,
        repo: &'static str,
        rev: &'static str,
        content_hash: &'static str,
        patch_name: &'static str,
    ) -> GitRepo {
        GitRepo { url: GitRepoUrl::Github { user, repo }, rev, content_hash, patch_name }
    }

    fn download_dir(&self, dirs: &Dirs) -> PathBuf {
        match self.url {
            GitRepoUrl::Github { user: _, repo } => RelPath::DOWNLOAD.join(repo).to_path(dirs),
        }
    }

    pub(crate) const fn source_dir(&self) -> RelPath {
        match self.url {
            GitRepoUrl::Github { user: _, repo } => RelPath::BUILD.join(repo),
        }
    }

    pub(crate) fn fetch(&self, dirs: &Dirs) {
        let download_dir = self.download_dir(dirs);

        if download_dir.exists() {
            let actual_hash = format!("{:016x}", hash_dir(&download_dir));
            if actual_hash == self.content_hash {
                println!("[FRESH] {}", download_dir.display());
                return;
            } else {
                println!(
                    "Mismatched content hash for {download_dir}: {actual_hash} != {content_hash}. Downloading again.",
                    download_dir = download_dir.display(),
                    content_hash = self.content_hash,
                );
            }
        }

        match self.url {
            GitRepoUrl::Github { user, repo } => {
                clone_repo_shallow_github(dirs, &download_dir, user, repo, self.rev);
            }
        }

        let source_lockfile =
            RelPath::PATCHES.to_path(dirs).join(format!("{}-lock.toml", self.patch_name));
        let target_lockfile = download_dir.join("Cargo.lock");
        if source_lockfile.exists() {
            fs::copy(source_lockfile, target_lockfile).unwrap();
        } else {
            assert!(target_lockfile.exists());
        }

        let actual_hash = format!("{:016x}", hash_dir(&download_dir));
        if actual_hash != self.content_hash {
            println!(
                "Download of {download_dir} failed with mismatched content hash: {actual_hash} != {content_hash}",
                download_dir = download_dir.display(),
                content_hash = self.content_hash,
            );
            std::process::exit(1);
        }
    }

    pub(crate) fn patch(&self, dirs: &Dirs) {
        apply_patches(
            dirs,
            self.patch_name,
            &self.download_dir(dirs),
            &self.source_dir().to_path(dirs),
        );
    }
}

#[allow(dead_code)]
fn clone_repo(download_dir: &Path, repo: &str, rev: &str) {
    eprintln!("[CLONE] {}", repo);
    // Ignore exit code as the repo may already have been checked out
    git_command(None, "clone").arg(repo).arg(download_dir).spawn().unwrap().wait().unwrap();

    let mut clean_cmd = git_command(download_dir, "checkout");
    clean_cmd.arg("--").arg(".");
    spawn_and_wait(clean_cmd);

    let mut checkout_cmd = git_command(download_dir, "checkout");
    checkout_cmd.arg("-q").arg(rev);
    spawn_and_wait(checkout_cmd);

    std::fs::remove_dir_all(download_dir.join(".git")).unwrap();
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
    download_cmd
        .arg("--max-time")
        .arg("600")
        .arg("-y")
        .arg("30")
        .arg("-Y")
        .arg("10")
        .arg("--connect-timeout")
        .arg("30")
        .arg("--continue-at")
        .arg("-")
        .arg("--location")
        .arg("--output")
        .arg(&archive_file)
        .arg(archive_url);
    retry_spawn_and_wait(5, download_cmd);

    // Unpack tar archive
    let mut unpack_cmd = Command::new("tar");
    unpack_cmd.arg("xf").arg(&archive_file).current_dir(RelPath::DOWNLOAD.to_path(dirs));
    spawn_and_wait(unpack_cmd);

    // Rename unpacked dir to the expected name
    std::fs::rename(archive_dir, &download_dir).unwrap();

    // Cleanup
    std::fs::remove_file(archive_file).unwrap();
}

fn init_git_repo(repo_dir: &Path) {
    let mut git_init_cmd = git_command(repo_dir, "init");
    git_init_cmd.arg("-q");
    spawn_and_wait(git_init_cmd);

    let mut git_add_cmd = git_command(repo_dir, "add");
    git_add_cmd.arg(".");
    spawn_and_wait(git_add_cmd);

    let mut git_commit_cmd = git_command(repo_dir, "commit");
    git_commit_cmd.arg("-m").arg("Initial commit").arg("-q");
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

pub(crate) fn apply_patches(dirs: &Dirs, crate_name: &str, source_dir: &Path, target_dir: &Path) {
    // FIXME avoid copy and patch if src, patches and target are unchanged

    eprintln!("[COPY] {crate_name} source");

    remove_dir_if_exists(target_dir);
    fs::create_dir_all(target_dir).unwrap();
    if crate_name == "stdlib" {
        fs::create_dir(target_dir.join("library")).unwrap();
        copy_dir_recursively(&source_dir.join("library"), &target_dir.join("library"));
    } else {
        copy_dir_recursively(source_dir, target_dir);
    }

    init_git_repo(target_dir);

    if crate_name == "<none>" {
        return;
    }

    for patch in get_patches(dirs, crate_name) {
        eprintln!(
            "[PATCH] {:?} <- {:?}",
            target_dir.file_name().unwrap(),
            patch.file_name().unwrap()
        );
        let mut apply_patch_cmd = git_command(target_dir, "am");
        apply_patch_cmd.arg(patch).arg("-q");
        spawn_and_wait(apply_patch_cmd);
    }
}
