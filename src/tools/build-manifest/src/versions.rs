use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::Error;
use flate2::read::GzDecoder;
use tar::Archive;
use xz2::read::XzDecoder;

const DEFAULT_TARGET: &str = "x86_64-unknown-linux-gnu";

macro_rules! pkg_type {
    ( $($variant:ident = $component:literal $(; preview = true $(@$is_preview:tt)? )? ),+ $(,)? ) => {
        #[derive(Debug, Hash, Eq, PartialEq, Clone)]
        pub(crate) enum PkgType {
            $($variant,)+
        }

        impl PkgType {
            pub(crate) fn is_preview(&self) -> bool {
                match self {
                    $( $( $($is_preview)? PkgType::$variant => true, )? )+
                    _ => false,
                }
            }

            /// First part of the tarball name.
            pub(crate) fn tarball_component_name(&self) -> &str {
                match self {
                    $( PkgType::$variant => $component,)+
                }
            }

            pub(crate) fn all() -> &'static [PkgType] {
                &[ $(PkgType::$variant),+ ]
            }
        }
    }
}

pkg_type! {
    Rust = "rust",
    RustSrc = "rust-src",
    Rustc = "rustc",
    RustcDev = "rustc-dev",
    RustcDocs = "rustc-docs",
    ReproducibleArtifacts = "reproducible-artifacts",
    RustMingw = "rust-mingw",
    RustStd = "rust-std",
    Cargo = "cargo",
    HtmlDocs = "rust-docs",
    RustAnalysis = "rust-analysis",
    RustAnalyzer = "rust-analyzer"; preview = true,
    Clippy = "clippy"; preview = true,
    Rustfmt = "rustfmt"; preview = true,
    LlvmTools = "llvm-tools"; preview = true,
    Miri = "miri"; preview = true,
    JsonDocs = "rust-docs-json"; preview = true,
    RustcCodegenCranelift = "rustc-codegen-cranelift"; preview = true,
    LlvmBitcodeLinker = "llvm-bitcode-linker"; preview = true,
}

impl PkgType {
    /// Component name in the manifest. In particular, this includes the `-preview` suffix where appropriate.
    pub(crate) fn manifest_component_name(&self) -> String {
        if self.is_preview() {
            format!("{}-preview", self.tarball_component_name())
        } else {
            self.tarball_component_name().to_string()
        }
    }

    /// Whether this package has the same version as Rust itself, or has its own `version` and
    /// `git-commit-hash` files inside the tarball.
    fn should_use_rust_version(&self) -> bool {
        match self {
            PkgType::Cargo => false,
            PkgType::RustAnalyzer => false,
            PkgType::Clippy => false,
            PkgType::Rustfmt => false,
            PkgType::LlvmTools => false,
            PkgType::Miri => false,
            PkgType::RustcCodegenCranelift => false,

            PkgType::Rust => true,
            PkgType::RustStd => true,
            PkgType::RustSrc => true,
            PkgType::Rustc => true,
            PkgType::JsonDocs => true,
            PkgType::HtmlDocs => true,
            PkgType::RustcDev => true,
            PkgType::RustcDocs => true,
            PkgType::ReproducibleArtifacts => true,
            PkgType::RustMingw => true,
            PkgType::RustAnalysis => true,
            PkgType::LlvmBitcodeLinker => true,
        }
    }

    pub(crate) fn targets(&self) -> &[&str] {
        use PkgType::*;

        use crate::{HOSTS, MINGW, TARGETS};

        match self {
            Rust => HOSTS, // doesn't matter in practice, but return something to avoid panicking
            Rustc => HOSTS,
            RustcDev => HOSTS,
            ReproducibleArtifacts => HOSTS,
            RustcDocs => HOSTS,
            Cargo => HOSTS,
            RustcCodegenCranelift => HOSTS,
            RustMingw => MINGW,
            RustStd => TARGETS,
            HtmlDocs => HOSTS,
            JsonDocs => HOSTS,
            RustSrc => &["*"],
            RustAnalyzer => HOSTS,
            Clippy => HOSTS,
            Miri => HOSTS,
            Rustfmt => HOSTS,
            RustAnalysis => TARGETS,
            LlvmTools => TARGETS,
            LlvmBitcodeLinker => HOSTS,
        }
    }

    /// Whether this package is target-independent or not.
    fn target_independent(&self) -> bool {
        *self == PkgType::RustSrc
    }

    /// Whether to package these target-specific docs for another similar target.
    pub(crate) fn use_docs_fallback(&self) -> bool {
        match self {
            PkgType::JsonDocs | PkgType::HtmlDocs => true,
            _ => false,
        }
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
                if *package == PkgType::Rust && version_info.version.is_none() {
                    panic!("missing version info for toolchain");
                }
                self.versions.insert(package.clone(), version_info.clone());
                Ok(version_info)
            }
        }
    }

    fn load_version_from_tarball(&mut self, package: &PkgType) -> Result<VersionInfo, Error> {
        for ext in ["xz", "gz"] {
            let info =
                self.load_version_from_tarball_inner(&self.dist_path.join(self.archive_name(
                    package,
                    DEFAULT_TARGET,
                    &format!("tar.{}", ext),
                )?))?;
            if info.present {
                return Ok(info);
            }
        }

        // If neither tarball is present, we fallback to returning the non-present info.
        Ok(VersionInfo::default())
    }

    fn load_version_from_tarball_inner(&mut self, tarball: &Path) -> Result<VersionInfo, Error> {
        let file = match File::open(&tarball) {
            Ok(file) => file,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                // Missing tarballs do not return an error, but return empty data.
                println!("warning: missing tarball {}", tarball.display());
                return Ok(VersionInfo::default());
            }
            Err(err) => return Err(err.into()),
        };
        let mut tar: Archive<Box<dyn std::io::Read>> =
            Archive::new(if tarball.extension().map_or(false, |e| e == "gz") {
                Box::new(GzDecoder::new(file))
            } else if tarball.extension().map_or(false, |e| e == "xz") {
                Box::new(XzDecoder::new(file))
            } else {
                unimplemented!("tarball extension not recognized: {}", tarball.display())
            });

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

    pub(crate) fn archive_name(
        &self,
        package: &PkgType,
        target: &str,
        extension: &str,
    ) -> Result<String, Error> {
        let component_name = package.tarball_component_name();
        let version = match self.channel.as_str() {
            "stable" => self.rustc_version().into(),
            "beta" => "beta".into(),
            "nightly" => "nightly".into(),
            _ => format!("{}-dev", self.rustc_version()),
        };

        if package.target_independent() {
            Ok(format!("{}-{}.{}", component_name, version, extension))
        } else {
            Ok(format!("{}-{}-{}.{}", component_name, version, target, extension))
        }
    }

    pub(crate) fn tarball_name(&self, package: &PkgType, target: &str) -> Result<String, Error> {
        self.archive_name(package, target, "tar.gz")
    }

    pub(crate) fn rustc_version(&self) -> &str {
        const RUSTC_VERSION: &str = include_str!("../../../version");
        RUSTC_VERSION.trim()
    }
}
