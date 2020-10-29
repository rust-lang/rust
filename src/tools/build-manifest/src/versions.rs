use anyhow::Error;
use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use tar::Archive;

const DEFAULT_TARGET: &str = "x86_64-unknown-linux-gnu";
const RUSTC_VERSION: &str = include_str!("../../../version");

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub(crate) enum PkgType {
    Rust,
    RustSrc,
    Rustc,
    Cargo,
    Rls,
    RustAnalyzer,
    Clippy,
    Rustfmt,
    LlvmTools,
    Miri,
    Other(String),
}

impl PkgType {
    pub(crate) fn from_component(component: &str) -> Self {
        match component {
            "rust" => PkgType::Rust,
            "rust-src" => PkgType::RustSrc,
            "rustc" => PkgType::Rustc,
            "cargo" => PkgType::Cargo,
            "rls" | "rls-preview" => PkgType::Rls,
            "rust-analyzer" | "rust-analyzer-preview" => PkgType::RustAnalyzer,
            "clippy" | "clippy-preview" => PkgType::Clippy,
            "rustfmt" | "rustfmt-preview" => PkgType::Rustfmt,
            "llvm-tools" | "llvm-tools-preview" => PkgType::LlvmTools,
            "miri" | "miri-preview" => PkgType::Miri,
            other => PkgType::Other(other.into()),
        }
    }

    /// First part of the tarball name.
    fn tarball_component_name(&self) -> &str {
        match self {
            PkgType::Rust => "rust",
            PkgType::RustSrc => "rust-src",
            PkgType::Rustc => "rustc",
            PkgType::Cargo => "cargo",
            PkgType::Rls => "rls",
            PkgType::RustAnalyzer => "rust-analyzer",
            PkgType::Clippy => "clippy",
            PkgType::Rustfmt => "rustfmt",
            PkgType::LlvmTools => "llvm-tools",
            PkgType::Miri => "miri",
            PkgType::Other(component) => component,
        }
    }

    /// Whether this package has the same version as Rust itself, or has its own `version` and
    /// `git-commit-hash` files inside the tarball.
    fn should_use_rust_version(&self) -> bool {
        match self {
            PkgType::Cargo => false,
            PkgType::Rls => false,
            PkgType::RustAnalyzer => false,
            PkgType::Clippy => false,
            PkgType::Rustfmt => false,
            PkgType::LlvmTools => false,
            PkgType::Miri => false,

            PkgType::Rust => true,
            PkgType::RustSrc => true,
            PkgType::Rustc => true,
            PkgType::Other(_) => true,
        }
    }

    /// Whether this package is target-independent or not.
    fn target_independent(&self) -> bool {
        *self == PkgType::RustSrc
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct VersionInfo {
    pub(crate) version: Option<String>,
    pub(crate) git_commit: Option<String>,
    pub(crate) present: bool,
}

pub(crate) struct Versions {
    channel: String,
    dist_path: PathBuf,
    versions: HashMap<PkgType, VersionInfo>,
}

impl Versions {
    pub(crate) fn new(channel: &str, dist_path: &Path) -> Result<Self, Error> {
        Ok(Self { channel: channel.into(), dist_path: dist_path.into(), versions: HashMap::new() })
    }

    pub(crate) fn channel(&self) -> &str {
        &self.channel
    }

    pub(crate) fn version(&mut self, mut package: &PkgType) -> Result<VersionInfo, Error> {
        if package.should_use_rust_version() {
            package = &PkgType::Rust;
        }

        match self.versions.get(package) {
            Some(version) => Ok(version.clone()),
            None => {
                let version_info = self.load_version_from_tarball(package)?;
                self.versions.insert(package.clone(), version_info.clone());
                Ok(version_info)
            }
        }
    }

    fn load_version_from_tarball(&mut self, package: &PkgType) -> Result<VersionInfo, Error> {
        let tarball_name = self.tarball_name(package, DEFAULT_TARGET)?;
        let tarball = self.dist_path.join(tarball_name);

        let file = match File::open(&tarball) {
            Ok(file) => file,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                // Missing tarballs do not return an error, but return empty data.
                return Ok(VersionInfo::default());
            }
            Err(err) => return Err(err.into()),
        };
        let mut tar = Archive::new(GzDecoder::new(file));

        let mut version = None;
        let mut git_commit = None;
        for entry in tar.entries()? {
            let mut entry = entry?;

            let dest;
            match entry.path()?.components().nth(1).and_then(|c| c.as_os_str().to_str()) {
                Some("version") => dest = &mut version,
                Some("git-commit-hash") => dest = &mut git_commit,
                _ => continue,
            }
            let mut buf = String::new();
            entry.read_to_string(&mut buf)?;
            *dest = Some(buf);

            // Short circuit to avoid reading the whole tar file if not necessary.
            if version.is_some() && git_commit.is_some() {
                break;
            }
        }

        Ok(VersionInfo { version, git_commit, present: true })
    }

    pub(crate) fn disable_version(&mut self, package: &PkgType) {
        match self.versions.get_mut(package) {
            Some(version) => {
                *version = VersionInfo::default();
            }
            None => {
                self.versions.insert(package.clone(), VersionInfo::default());
            }
        }
    }

    pub(crate) fn archive_name(
        &mut self,
        package: &PkgType,
        target: &str,
        extension: &str,
    ) -> Result<String, Error> {
        let component_name = package.tarball_component_name();
        let version = match self.channel.as_str() {
            "stable" => RUSTC_VERSION.into(),
            "beta" => "beta".into(),
            "nightly" => "nightly".into(),
            _ => format!("{}-dev", RUSTC_VERSION),
        };

        if package.target_independent() {
            Ok(format!("{}-{}.{}", component_name, version, extension))
        } else {
            Ok(format!("{}-{}-{}.{}", component_name, version, target, extension))
        }
    }

    pub(crate) fn tarball_name(
        &mut self,
        package: &PkgType,
        target: &str,
    ) -> Result<String, Error> {
        self.archive_name(package, target, "tar.gz")
    }

    pub(crate) fn rustc_version(&self) -> &str {
        RUSTC_VERSION
    }
}
