//! FIXME: write short doc here

mod cargo_workspace;
mod cfg_flag;
mod project_json;
mod sysroot;
mod workspace;
mod rustc_cfg;
mod build_data;

use std::{
    fs::{read_dir, ReadDir},
    io,
    process::Command,
};

use anyhow::{bail, Context, Result};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashSet;

pub use crate::{
    build_data::{BuildDataCollector, BuildDataResult},
    cargo_workspace::{
        CargoConfig, CargoWorkspace, Package, PackageData, PackageDependency, RustcSource, Target,
        TargetData, TargetKind,
    },
    project_json::{ProjectJson, ProjectJsonData},
    sysroot::Sysroot,
    workspace::{PackageRoot, ProjectWorkspace},
};

pub use proc_macro_api::ProcMacroClient;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum ProjectManifest {
    ProjectJson(AbsPathBuf),
    CargoToml(AbsPathBuf),
}

impl ProjectManifest {
    pub fn from_manifest_file(path: AbsPathBuf) -> Result<ProjectManifest> {
        if path.ends_with("rust-project.json") {
            return Ok(ProjectManifest::ProjectJson(path));
        }
        if path.ends_with("Cargo.toml") {
            return Ok(ProjectManifest::CargoToml(path));
        }
        bail!("project root must point to Cargo.toml or rust-project.json: {}", path.display())
    }

    pub fn discover_single(path: &AbsPath) -> Result<ProjectManifest> {
        let mut candidates = ProjectManifest::discover(path)?;
        let res = match candidates.pop() {
            None => bail!("no projects"),
            Some(it) => it,
        };

        if !candidates.is_empty() {
            bail!("more than one project")
        }
        Ok(res)
    }

    pub fn discover(path: &AbsPath) -> io::Result<Vec<ProjectManifest>> {
        if let Some(project_json) = find_in_parent_dirs(path, "rust-project.json") {
            return Ok(vec![ProjectManifest::ProjectJson(project_json)]);
        }
        return find_cargo_toml(path)
            .map(|paths| paths.into_iter().map(ProjectManifest::CargoToml).collect());

        fn find_cargo_toml(path: &AbsPath) -> io::Result<Vec<AbsPathBuf>> {
            match find_in_parent_dirs(path, "Cargo.toml") {
                Some(it) => Ok(vec![it]),
                None => Ok(find_cargo_toml_in_child_dir(read_dir(path)?)),
            }
        }

        fn find_in_parent_dirs(path: &AbsPath, target_file_name: &str) -> Option<AbsPathBuf> {
            if path.ends_with(target_file_name) {
                return Some(path.to_path_buf());
            }

            let mut curr = Some(path);

            while let Some(path) = curr {
                let candidate = path.join(target_file_name);
                if candidate.exists() {
                    return Some(candidate);
                }
                curr = path.parent();
            }

            None
        }

        fn find_cargo_toml_in_child_dir(entities: ReadDir) -> Vec<AbsPathBuf> {
            // Only one level down to avoid cycles the easy way and stop a runaway scan with large projects
            entities
                .filter_map(Result::ok)
                .map(|it| it.path().join("Cargo.toml"))
                .filter(|it| it.exists())
                .map(AbsPathBuf::assert)
                .collect()
        }
    }

    pub fn discover_all(paths: &[impl AsRef<AbsPath>]) -> Vec<ProjectManifest> {
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
    let output = cmd.output().with_context(|| format!("{:?} failed", cmd))?;
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
