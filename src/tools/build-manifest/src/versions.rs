use anyhow::{Context, Error};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub(crate) enum PkgType {
    Rust,
    RustSrc,
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

    fn rust_monorepo_path(&self) -> Option<&'static str> {
        match self {
            PkgType::Cargo => Some("src/tools/cargo"),
            PkgType::Rls => Some("src/tools/rls"),
            PkgType::RustAnalyzer => Some("src/tools/rust-analyzer/crates/rust-analyzer"),
            PkgType::Clippy => Some("src/tools/clippy"),
            PkgType::Rustfmt => Some("src/tools/rustfmt"),
            PkgType::Miri => Some("src/tools/miri"),
            PkgType::Rust => None,
            PkgType::RustSrc => None,
            PkgType::LlvmTools => None,
            PkgType::Other(_) => None,
        }
    }

    fn tarball_component_name(&self) -> &str {
        match self {
            PkgType::Rust => "rust",
            PkgType::RustSrc => "rust-src",
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
}

pub(crate) struct Versions {
    channel: String,
    rustc_version: String,
    monorepo_root: PathBuf,
    package_versions: HashMap<PkgType, String>,
}

impl Versions {
    pub(crate) fn new(channel: &str, monorepo_root: &Path) -> Result<Self, Error> {
        Ok(Self {
            channel: channel.into(),
            rustc_version: std::fs::read_to_string(monorepo_root.join("src").join("version"))
                .context("failed to read the rustc version from src/version")?
                .trim()
                .to_string(),
            monorepo_root: monorepo_root.into(),
            package_versions: HashMap::new(),
        })
    }

    pub(crate) fn channel(&self) -> &str {
        &self.channel
    }

    pub(crate) fn tarball_name(
        &mut self,
        package: &PkgType,
        target: &str,
    ) -> Result<String, Error> {
        Ok(format!(
            "{}-{}-{}.tar.gz",
            package.tarball_component_name(),
            self.package_version(package).with_context(|| format!(
                "failed to get the package version for component {:?}",
                package,
            ))?,
            target
        ))
    }

    pub(crate) fn package_version(&mut self, package: &PkgType) -> Result<String, Error> {
        match self.package_versions.get(package) {
            Some(release) => Ok(release.clone()),
            None => {
                let version = match package.rust_monorepo_path() {
                    Some(path) => {
                        let path = self.monorepo_root.join(path).join("Cargo.toml");
                        let cargo_toml: CargoToml = toml::from_slice(&std::fs::read(path)?)?;
                        cargo_toml.package.version
                    }
                    None => self.rustc_version.clone(),
                };

                let release = match self.channel.as_str() {
                    "stable" => version,
                    "beta" => "beta".into(),
                    "nightly" => "nightly".into(),
                    _ => format!("{}-dev", version),
                };

                self.package_versions.insert(package.clone(), release.clone());
                Ok(release)
            }
        }
    }
}

#[derive(serde::Deserialize)]
struct CargoToml {
    package: CargoTomlPackage,
}

#[derive(serde::Deserialize)]
struct CargoTomlPackage {
    version: String,
}
