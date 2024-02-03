//! This module contains the implementation for generating rust-project.json data which can be
//! utilized for LSPs (Language Server Protocols).
//!
//! The primary reason for relying on rust-analyzer.json instead of the default rust-analyzer
//! is because rust-analyzer is not so capable of handling rust-lang/rust workspaces out of the box.
//! It often encounters new issues while trying to fix current problems with some hacky workarounds.
//!
//! For additional context, see the [zulip thread].
//!
//! [zulip thread]: https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler/topic/r-a.20support.20for.20rust-lang.2Frust.20via.20project-rust.2Ejson/near/412505824

use serde_derive::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::Path;

use crate::core::metadata::{project_metadata, workspace_members, Dependency};
use crate::Config;

#[derive(Debug, Serialize)]
/// FIXME(before-merge): doc-comment
pub(crate) struct RustAnalyzerProject {
    crates: Vec<Crate>,
    sysroot: String,
    sysroot_src: String,
}

#[derive(Debug, Default, Serialize, PartialEq)]
struct Crate {
    cfg: Vec<String>,
    deps: BTreeSet<Dep>,
    display_name: String,
    edition: String,
    env: BTreeMap<String, String>,
    is_proc_macro: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    proc_macro_dylib_path: Option<String>,
    is_workspace_member: bool,
    root_module: String,
}

#[derive(Debug, Default, Serialize, PartialEq, PartialOrd, Ord, Eq)]
struct Dep {
    #[serde(rename = "crate")]
    crate_index: usize,
    name: String,
}

impl RustAnalyzerProject {
    #[allow(dead_code)] // FIXME(before-merge): remove this
    pub(crate) fn collect_ra_project_data(config: &Config) -> Self {
        let mut ra_project = RustAnalyzerProject {
            crates: vec![],
            sysroot: format!("{}", config.out.join("host").join("stage0").display()),
            sysroot_src: format!("{}", config.src.join("library").display()),
        };

        let packages: Vec<_> = project_metadata(config).collect();
        let workspace_members: Vec<_> = workspace_members(config).collect();

        for package in &packages {
            let is_not_indirect_dependency = packages
                .iter()
                .filter(|t| {
                    let used_from_other_crates = t.dependencies.contains(&Dependency {
                        name: package.name.clone(),
                        source: package.source.clone(),
                    });

                    let is_local = t.source.is_none();

                    (used_from_other_crates && is_local) || package.source.is_none()
                })
                .next()
                .is_some();

            if !is_not_indirect_dependency {
                continue;
            }

            for target in &package.targets {
                let mut krate = Crate::default();
                krate.display_name = target.name.clone();
                krate.root_module = target.src_path.clone();
                krate.edition = target.edition.clone();
                krate.is_workspace_member = workspace_members.iter().any(|p| p.name == target.name);
                krate.is_proc_macro = target.crate_types.contains(&"proc-macro".to_string());

                // FIXME(before-merge): We need to figure out how to find proc-macro dylibs.
                // if krate.is_proc_macro {
                //     krate.proc_macro_dylib_path =
                // }

                krate.env.insert("RUSTC_BOOTSTRAP".into(), "1".into());

                if target
                    .src_path
                    .starts_with(&config.src.join("library").to_string_lossy().to_string())
                {
                    krate.cfg.push("bootstrap".into());
                }

                ra_project.crates.push(krate);
            }
        }

        ra_project.crates.sort_by_key(|c| c.display_name.clone());
        ra_project.crates.dedup_by_key(|c| c.display_name.clone());

        // Find and fill dependencies of crates.
        for package in packages {
            if package.dependencies.is_empty() {
                continue;
            }

            for dependency in package.dependencies {
                if let Some(index) =
                    ra_project.crates.iter().position(|c| c.display_name == package.name)
                {
                    if let Some(dependency_index) =
                        ra_project.crates.iter().position(|c| c.display_name == dependency.name)
                    {
                        // no need to find indirect dependencies of direct dependencies, just continue
                        if ra_project.crates[index].root_module.contains(".cargo/registry") {
                            continue;
                        }

                        let dependency_name = dependency.name.replace('-', "_").to_lowercase();

                        ra_project.crates[index]
                            .deps
                            .insert(Dep { crate_index: dependency_index, name: dependency_name });
                    }
                }
            }
        }

        ra_project
    }

    #[allow(dead_code)] // FIXME(before-merge): remove this
    pub(crate) fn generate_file(&self, path: &Path) -> io::Result<()> {
        if path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("File '{}' already exists.", path.display()),
            ));
        }

        let mut file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(&mut file, self)?;

        Ok(())
    }
}
