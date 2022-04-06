use dirs;
use std::{fmt, path::PathBuf, process::Command};

use super::version_manager::{ENZYME_VER, RUSTC_VER};

use clap::Parser;

/// A struct used to manage which Enzyme / Rustc combination should be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cli {
    /// Select the rust version to build and use
    pub rust: Repo,
    /// Select the enzyme version to build and use
    pub enzyme: Repo,
}

impl Cli {
    /// Parses terminal input into a Cli struct.
    pub fn parse() -> Cli {
        App::parse().into()
    }
}

impl From<App> for Cli {
    fn from(app: App) -> Self {
        let rust: Repo = if let Some(local) = app.rust_local.as_deref() {
            let p = PathBuf::from(local.to_string());
            assert!(p.is_dir());
            Repo::Local(p)
        } else if app.rust_stable {
            Repo::Stable
        } else if app.rust_head {
            Repo::Head
        } else {
            unreachable!()
        };
        let enzyme: Repo = if let Some(local) = app.enzyme_local.as_deref() {
            let p = PathBuf::from(local.to_string());
            assert!(p.is_dir());
            Repo::Local(p)
        } else if app.enzyme_stable {
            Repo::Stable
        } else if app.enzyme_head {
            Repo::Head
        } else {
            unreachable!()
        };
        Cli { rust, enzyme }
    }
}

/// Used to decide which Enzyme Version should be used
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Repo {
    /// Use the latest Enzyme stable release
    Stable,
    /// Use the current Enzyme head from Github
    Head,
    /// Use a local Enzyme repository at the given path
    Local(PathBuf),
}

#[derive(Parser)]
#[clap(
    group(clap::ArgGroup::new("rust").required(true).args(&["rust-stable", "rust-head", "rust-local"])),
    group(clap::ArgGroup::new("enzyme").required(true).args(&["enzyme-stable", "enzyme-head", "enzyme-local"])),
    author, version, about, long_about = None
    )]
struct App {
    /// Use the latest official Rust release
    #[clap(long = "rust-stable")]
    rust_stable: bool,
    /// Use the master branch from https://github.com/rust-lang/rust/
    #[clap(long = "rust-head")]
    rust_head: bool,
    /// Use the (Rust) repository at the given path
    #[clap(long)]
    rust_local: Option<String>,
    /// Use the latest official Enzyme release
    #[clap(long = "enzyme-stable")]
    enzyme_stable: bool,
    #[clap(long = "enzyme-head")]
    /// Use the main branch from https://github.com/EnzymeAD/Enzyme
    enzyme_head: bool,
    /// Use the (Enzyme) repository at the given path
    #[clap(long)]
    enzyme_local: Option<String>,
}

pub(crate) fn run_and_printerror(command: &mut Command) {
    println!("Running: `{:?}`", command);
    match command.status() {
        Ok(status) => {
            if !status.success() {
                panic!("Failed: `{:?}` ({})", command, status);
            }
        }
        Err(error) => {
            panic!("Failed: `{:?}` ({})", command, error);
        }
    }
}

fn assert_existence(path: PathBuf) {
    if !path.is_dir() {
        std::fs::create_dir_all(path.clone())
            .unwrap_or_else(|_| panic!("Couldn't create: {}", path.display()));
    }
}
pub fn get_enzyme_base_path() -> PathBuf {
    let cache_dir = dirs::cache_dir().expect("Enzyme needs access to your cache dir.");
    let enzyme_base_path = cache_dir.join("enzyme");
    assert_existence(enzyme_base_path.clone());
    enzyme_base_path
}

// Following is a list of repo locations.
pub fn get_local_repo_dir(repo: Repo, which: Selection) -> PathBuf {
    match which {
        Selection::Rust => get_local_rust_repo_path(repo),
        Selection::Enzyme => get_local_enzyme_repo_path(repo),
    }
}
pub fn get_local_rust_repo_path(rust: Repo) -> PathBuf {
    match rust {
        Repo::Stable => get_stable_repo_dir(Selection::Rust),
        Repo::Head => get_head_repo_dir(Selection::Rust),
        Repo::Local(l) => l,
    }
}
pub fn get_local_enzyme_repo_path(enzyme: Repo) -> PathBuf {
    match enzyme {
        Repo::Stable => get_stable_repo_dir(Selection::Enzyme),
        Repo::Head => get_head_repo_dir(Selection::Enzyme),
        Repo::Local(l) => l,
    }
}
pub fn get_remote_repo_url(which: Selection) -> String {
    match which {
        Selection::Enzyme => "https://github.com/EnzymeAD/Enzyme".to_string(),
        Selection::Rust => "https://github.com/rust-lang/rust".to_string(),
    }
}
pub fn get_stable_repo_dir(which: Selection) -> PathBuf {
    let subdir = match which {
        Selection::Enzyme => "Enzyme-".to_owned() + ENZYME_VER,
        Selection::Rust => "rustc-".to_owned() + RUSTC_VER + "-src",
    };
    let path = get_enzyme_base_path().join(subdir);
    assert_existence(path.clone());
    path
}
pub fn get_head_repo_dir(which: Selection) -> PathBuf {
    let path = match which {
        Selection::Rust => get_enzyme_base_path().join("rustc-HEAD-src"),
        Selection::Enzyme => get_enzyme_base_path().join("Enzyme-HEAD"),
    };
    assert_existence(path.clone());
    path
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Selection {
    Rust,
    Enzyme,
}
impl fmt::Display for Selection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Selection::Rust => write!(f, "Rust"),
            Selection::Enzyme => write!(f, "Enzyme"),
        }
    }
}

pub fn get_local_tarball_path(which: Selection) -> PathBuf {
    match which {
        Selection::Rust => get_download_dir().join("rustc-".to_owned() + RUSTC_VER + ".tar.gz"),
        Selection::Enzyme => get_download_dir().join("enzyme-".to_owned() + ENZYME_VER + ".tar.gz"),
    }
}
pub fn get_remote_tarball_url(which: Selection) -> String {
    match which {
        Selection::Rust => format!(
            "https://static.rust-lang.org/dist/rustc-{}-src.tar.gz",
            RUSTC_VER
        ),
        Selection::Enzyme => format!(
            "https://github.com/EnzymeAD/Enzyme/archive/refs/tags/v{}.tar.gz",
            ENZYME_VER
        ),
    }
}

/// Returns the path to the generated binding file.
pub fn get_bindings_path() -> PathBuf {
    get_enzyme_base_path().join("enzyme.rs")
}
// Following is a list of function used internally
fn get_enzyme_subdir_path(enzyme_repo_path: PathBuf) -> PathBuf {
    enzyme_repo_path.join("enzyme")
}
pub fn get_capi_path(enzyme_repo_path: PathBuf) -> PathBuf {
    get_enzyme_subdir_path(enzyme_repo_path)
        .join("Enzyme")
        .join("CApi.h")
}
pub fn get_enzyme_build_path(repo_path: PathBuf) -> PathBuf {
    let enzyme_path = get_enzyme_subdir_path(repo_path).join("build");
    assert_existence(enzyme_path.clone());
    enzyme_path
}
pub fn get_download_dir() -> PathBuf {
    let enzyme_download_path = get_enzyme_base_path().join("downloads");
    assert_existence(enzyme_download_path.clone());
    enzyme_download_path
}
pub fn get_rustc_build_path(rust_repo: PathBuf) -> PathBuf {
    let rustc_path = rust_repo.join("build");
    assert_existence(rustc_path.clone());
    rustc_path
}
fn get_rustc_platform_path(rust_repo: PathBuf) -> PathBuf {
    let platform = env!("TARGET");
    get_rustc_build_path(rust_repo).join(&platform)
}
pub fn get_rustc_stage2_path(repo_path: PathBuf) -> PathBuf {
    get_rustc_platform_path(repo_path).join("stage2")
}
pub fn get_llvm_build_path(rust_repo: PathBuf) -> PathBuf {
    get_rustc_platform_path(rust_repo)
        .join("llvm")
        .join("build")
}
pub fn get_llvm_header_path(rust_repo: PathBuf) -> PathBuf {
    get_rustc_platform_path(rust_repo)
        .join("llvm")
        .join("include")
}
