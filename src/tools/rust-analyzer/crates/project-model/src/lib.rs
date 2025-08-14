//! In rust-analyzer, we maintain a strict separation between pure abstract
//! semantic project model and a concrete model of a particular build system.
//!
//! Pure model is represented by the [`base_db::CrateGraph`] from another crate.
//!
//! In this crate, we are concerned with "real world" project models.
//!
//! Specifically, here we have a representation for a Cargo project
//! ([`CargoWorkspace`]) and for manually specified layout ([`ProjectJson`]).
//!
//! Roughly, the things we do here are:
//!
//! * Project discovery (where's the relevant Cargo.toml for the current dir).
//! * Custom build steps (`build.rs` code generation and compilation of
//!   procedural macros).
//! * Lowering of concrete model to a [`base_db::CrateGraph`]

pub mod project_json;
pub mod toolchain_info {
    pub mod rustc_cfg;
    pub mod target_data_layout;
    pub mod target_tuple;
    pub mod version;

    use std::path::Path;

    use crate::{ManifestPath, Sysroot, cargo_config_file::CargoConfigFile};

    #[derive(Copy, Clone)]
    pub enum QueryConfig<'a> {
        /// Directly invoke `rustc` to query the desired information.
        Rustc(&'a Sysroot, &'a Path),
        /// Attempt to use cargo to query the desired information, honoring cargo configurations.
        /// If this fails, falls back to invoking `rustc` directly.
        Cargo(&'a Sysroot, &'a ManifestPath, &'a Option<CargoConfigFile>),
    }
}

mod build_dependencies;
mod cargo_config_file;
mod cargo_workspace;
mod env;
mod manifest_path;
mod sysroot;
mod workspace;

#[cfg(test)]
mod tests;

use std::{
    fmt,
    fs::{self, ReadDir, read_dir},
    io,
    process::Command,
};

use anyhow::{Context, bail, format_err};
use paths::{AbsPath, AbsPathBuf, Utf8PathBuf};
use rustc_hash::FxHashSet;

pub use crate::{
    build_dependencies::{ProcMacroDylibPath, WorkspaceBuildScripts},
    cargo_workspace::{
        CargoConfig, CargoFeatures, CargoMetadataConfig, CargoWorkspace, Package, PackageData,
        PackageDependency, RustLibSource, Target, TargetData, TargetKind,
    },
    manifest_path::ManifestPath,
    project_json::{ProjectJson, ProjectJsonData},
    sysroot::Sysroot,
    workspace::{FileLoader, PackageRoot, ProjectWorkspace, ProjectWorkspaceKind},
};
pub use cargo_metadata::Metadata;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectJsonFromCommand {
    /// The data describing this project, such as its dependencies.
    pub data: ProjectJsonData,
    /// The build system specific file that describes this project,
    /// such as a `my-project/BUCK` file.
    pub buildfile: AbsPathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum ProjectManifest {
    ProjectJson(ManifestPath),
    CargoToml(ManifestPath),
    CargoScript(ManifestPath),
}

impl ProjectManifest {
    pub fn from_manifest_file(path: AbsPathBuf) -> anyhow::Result<ProjectManifest> {
        let path = ManifestPath::try_from(path)
            .map_err(|path| format_err!("bad manifest path: {path}"))?;
        if path.file_name().unwrap_or_default() == "rust-project.json" {
            return Ok(ProjectManifest::ProjectJson(path));
        }
        if path.file_name().unwrap_or_default() == ".rust-project.json" {
            return Ok(ProjectManifest::ProjectJson(path));
        }
        if path.file_name().unwrap_or_default() == "Cargo.toml" {
            return Ok(ProjectManifest::CargoToml(path));
        }
        if path.extension().unwrap_or_default() == "rs" {
            return Ok(ProjectManifest::CargoScript(path));
        }
        bail!(
            "project root must point to a Cargo.toml, rust-project.json or <script>.rs file: {path}"
        );
    }

    pub fn discover_single(path: &AbsPath) -> anyhow::Result<ProjectManifest> {
        let mut candidates = ProjectManifest::discover(path)?;
        let res = match candidates.pop() {
            None => bail!("no projects"),
            Some(it) => it,
        };

        if !candidates.is_empty() {
            bail!("more than one project");
        }
        Ok(res)
    }

    pub fn discover(path: &AbsPath) -> io::Result<Vec<ProjectManifest>> {
        if let Some(project_json) = find_in_parent_dirs(path, "rust-project.json") {
            return Ok(vec![ProjectManifest::ProjectJson(project_json)]);
        }
        if let Some(project_json) = find_in_parent_dirs(path, ".rust-project.json") {
            return Ok(vec![ProjectManifest::ProjectJson(project_json)]);
        }
        return find_cargo_toml(path)
            .map(|paths| paths.into_iter().map(ProjectManifest::CargoToml).collect());

        fn find_cargo_toml(path: &AbsPath) -> io::Result<Vec<ManifestPath>> {
            match find_in_parent_dirs(path, "Cargo.toml") {
                Some(it) => Ok(vec![it]),
                None => Ok(find_cargo_toml_in_child_dir(read_dir(path)?)),
            }
        }

        fn find_in_parent_dirs(path: &AbsPath, target_file_name: &str) -> Option<ManifestPath> {
            if path.file_name().unwrap_or_default() == target_file_name
                && let Ok(manifest) = ManifestPath::try_from(path.to_path_buf())
            {
                return Some(manifest);
            }

            let mut curr = Some(path);

            while let Some(path) = curr {
                let candidate = path.join(target_file_name);
                if fs::metadata(&candidate).is_ok()
                    && let Ok(manifest) = ManifestPath::try_from(candidate)
                {
                    return Some(manifest);
                }

                curr = path.parent();
            }

            None
        }

        fn find_cargo_toml_in_child_dir(entities: ReadDir) -> Vec<ManifestPath> {
            // Only one level down to avoid cycles the easy way and stop a runaway scan with large projects
            entities
                .filter_map(Result::ok)
                .map(|it| it.path().join("Cargo.toml"))
                .filter(|it| it.exists())
                .map(Utf8PathBuf::from_path_buf)
                .filter_map(Result::ok)
                .map(AbsPathBuf::try_from)
                .filter_map(Result::ok)
                .filter_map(|it| it.try_into().ok())
                .collect()
        }
    }

    pub fn discover_all(paths: &[AbsPathBuf]) -> Vec<ProjectManifest> {
        let mut res = paths
            .iter()
            .filter_map(|it| ProjectManifest::discover(it.as_ref()).ok())
            .flatten()
            .collect::<FxHashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        res.sort();
        res
    }

    pub fn manifest_path(&self) -> &ManifestPath {
        match self {
            ProjectManifest::ProjectJson(it)
            | ProjectManifest::CargoToml(it)
            | ProjectManifest::CargoScript(it) => it,
        }
    }
}

impl fmt::Display for ProjectManifest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.manifest_path(), f)
    }
}

fn utf8_stdout(cmd: &mut Command) -> anyhow::Result<String> {
    let output = cmd.output().with_context(|| format!("{cmd:?} failed"))?;
    if !output.status.success() {
        match String::from_utf8(output.stderr) {
            Ok(stderr) if !stderr.is_empty() => {
                bail!("{:?} failed, {}\nstderr:\n{}", cmd, output.status, stderr)
            }
            _ => bail!("{:?} failed, {}", cmd, output.status),
        }
    }
    let stdout = String::from_utf8(output.stdout)?;
    Ok(stdout.trim().to_owned())
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum InvocationStrategy {
    Once,
    #[default]
    PerWorkspace,
}

/// A set of cfg-overrides per crate.
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct CfgOverrides {
    /// A global set of overrides matching all crates.
    pub global: cfg::CfgDiff,
    /// A set of overrides matching specific crates.
    pub selective: rustc_hash::FxHashMap<String, cfg::CfgDiff>,
}

impl CfgOverrides {
    pub fn len(&self) -> usize {
        self.global.len() + self.selective.values().map(|it| it.len()).sum::<usize>()
    }

    pub fn apply(&self, cfg_options: &mut cfg::CfgOptions, name: &str) {
        if !self.global.is_empty() {
            cfg_options.apply_diff(self.global.clone());
        };
        if let Some(diff) = self.selective.get(name) {
            cfg_options.apply_diff(diff.clone());
        };
    }
}

fn parse_cfg(s: &str) -> Result<cfg::CfgAtom, String> {
    let res = match s.split_once('=') {
        Some((key, value)) => {
            if !(value.starts_with('"') && value.ends_with('"')) {
                return Err(format!("Invalid cfg ({s:?}), value should be in quotes"));
            }
            let key = intern::Symbol::intern(key);
            let value = intern::Symbol::intern(&value[1..value.len() - 1]);
            cfg::CfgAtom::KeyValue { key, value }
        }
        None => cfg::CfgAtom::Flag(intern::Symbol::intern(s)),
    };
    Ok(res)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RustSourceWorkspaceConfig {
    CargoMetadata(CargoMetadataConfig),
    Json(ProjectJson),
}

impl Default for RustSourceWorkspaceConfig {
    fn default() -> Self {
        RustSourceWorkspaceConfig::default_cargo()
    }
}

impl RustSourceWorkspaceConfig {
    pub fn default_cargo() -> Self {
        RustSourceWorkspaceConfig::CargoMetadata(Default::default())
    }
}
