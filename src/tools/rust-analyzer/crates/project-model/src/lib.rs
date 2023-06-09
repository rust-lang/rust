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

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

mod manifest_path;
mod cargo_workspace;
mod cfg_flag;
mod project_json;
mod sysroot;
mod workspace;
mod rustc_cfg;
mod build_scripts;
pub mod target_data_layout;

#[cfg(test)]
mod tests;

use std::{
    fs::{self, read_dir, ReadDir},
    io,
    process::Command,
};

use anyhow::{bail, format_err, Context, Result};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashSet;

pub use crate::{
    build_scripts::WorkspaceBuildScripts,
    cargo_workspace::{
        CargoConfig, CargoFeatures, CargoWorkspace, Package, PackageData, PackageDependency,
        RustLibSource, Target, TargetData, TargetKind,
    },
    manifest_path::ManifestPath,
    project_json::{ProjectJson, ProjectJsonData},
    sysroot::Sysroot,
    workspace::{CfgOverrides, PackageRoot, ProjectWorkspace},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum ProjectManifest {
    ProjectJson(ManifestPath),
    CargoToml(ManifestPath),
}

impl ProjectManifest {
    pub fn from_manifest_file(path: AbsPathBuf) -> Result<ProjectManifest> {
        let path = ManifestPath::try_from(path)
            .map_err(|path| format_err!("bad manifest path: {}", path.display()))?;
        if path.file_name().unwrap_or_default() == "rust-project.json" {
            return Ok(ProjectManifest::ProjectJson(path));
        }
        if path.file_name().unwrap_or_default() == "Cargo.toml" {
            return Ok(ProjectManifest::CargoToml(path));
        }
        bail!("project root must point to Cargo.toml or rust-project.json: {}", path.display());
    }

    pub fn discover_single(path: &AbsPath) -> Result<ProjectManifest> {
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
        return find_cargo_toml(path)
            .map(|paths| paths.into_iter().map(ProjectManifest::CargoToml).collect());

        fn find_cargo_toml(path: &AbsPath) -> io::Result<Vec<ManifestPath>> {
            match find_in_parent_dirs(path, "Cargo.toml") {
                Some(it) => Ok(vec![it]),
                None => Ok(find_cargo_toml_in_child_dir(read_dir(path)?)),
            }
        }

        fn find_in_parent_dirs(path: &AbsPath, target_file_name: &str) -> Option<ManifestPath> {
            if path.file_name().unwrap_or_default() == target_file_name {
                if let Ok(manifest) = ManifestPath::try_from(path.to_path_buf()) {
                    return Some(manifest);
                }
            }

            let mut curr = Some(path);

            while let Some(path) = curr {
                let candidate = path.join(target_file_name);
                if fs::metadata(&candidate).is_ok() {
                    if let Ok(manifest) = ManifestPath::try_from(candidate) {
                        return Some(manifest);
                    }
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
                .map(AbsPathBuf::assert)
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
}

fn utf8_stdout(mut cmd: Command) -> Result<String> {
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
    Ok(stdout.trim().to_string())
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum InvocationStrategy {
    Once,
    #[default]
    PerWorkspace,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum InvocationLocation {
    Root(AbsPathBuf),
    #[default]
    Workspace,
}
