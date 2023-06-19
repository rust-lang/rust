//! Handles lowering of build-system specific workspace information (`cargo
//! metadata` or `rust-project.json`) into representation stored in the salsa
//! database -- `CrateGraph`.

use std::{collections::VecDeque, fmt, fs, process::Command, sync};

use anyhow::{format_err, Context, Result};
use base_db::{
    CrateDisplayName, CrateGraph, CrateId, CrateName, CrateOrigin, Dependency, Edition, Env,
    FileId, LangCrateOrigin, ProcMacroPaths, ReleaseChannel, TargetLayoutLoadResult,
};
use cfg::{CfgDiff, CfgOptions};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::{FxHashMap, FxHashSet};
use semver::Version;
use stdx::always;
use triomphe::Arc;

use crate::{
    build_scripts::BuildScriptOutput,
    cargo_workspace::{DepKind, PackageData, RustLibSource},
    cfg_flag::CfgFlag,
    project_json::Crate,
    rustc_cfg,
    sysroot::SysrootCrate,
    target_data_layout, utf8_stdout, CargoConfig, CargoWorkspace, InvocationStrategy, ManifestPath,
    Package, ProjectJson, ProjectManifest, Sysroot, TargetData, TargetKind, WorkspaceBuildScripts,
};

/// A set of cfg-overrides per crate.
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct CfgOverrides {
    /// A global set of overrides matching all crates.
    pub global: CfgDiff,
    /// A set of overrides matching specific crates.
    pub selective: FxHashMap<String, CfgDiff>,
}

impl CfgOverrides {
    pub fn len(&self) -> usize {
        self.global.len() + self.selective.iter().map(|(_, it)| it.len()).sum::<usize>()
    }
}

/// `PackageRoot` describes a package root folder.
/// Which may be an external dependency, or a member of
/// the current workspace.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct PackageRoot {
    /// Is from the local filesystem and may be edited
    pub is_local: bool,
    pub include: Vec<AbsPathBuf>,
    pub exclude: Vec<AbsPathBuf>,
}

#[derive(Clone)]
pub enum ProjectWorkspace {
    /// Project workspace was discovered by running `cargo metadata` and `rustc --print sysroot`.
    Cargo {
        cargo: CargoWorkspace,
        build_scripts: WorkspaceBuildScripts,
        sysroot: Result<Sysroot, Option<String>>,
        rustc: Result<(CargoWorkspace, WorkspaceBuildScripts), Option<String>>,
        /// Holds cfg flags for the current target. We get those by running
        /// `rustc --print cfg`.
        ///
        /// FIXME: make this a per-crate map, as, eg, build.rs might have a
        /// different target.
        rustc_cfg: Vec<CfgFlag>,
        cfg_overrides: CfgOverrides,
        toolchain: Option<Version>,
        target_layout: Result<String, String>,
    },
    /// Project workspace was manually specified using a `rust-project.json` file.
    Json {
        project: ProjectJson,
        sysroot: Result<Sysroot, Option<String>>,
        /// Holds cfg flags for the current target. We get those by running
        /// `rustc --print cfg`.
        rustc_cfg: Vec<CfgFlag>,
        toolchain: Option<Version>,
    },
    // FIXME: The primary limitation of this approach is that the set of detached files needs to be fixed at the beginning.
    // That's not the end user experience we should strive for.
    // Ideally, you should be able to just open a random detached file in existing cargo projects, and get the basic features working.
    // That needs some changes on the salsa-level though.
    // In particular, we should split the unified CrateGraph (which currently has maximal durability) into proper crate graph, and a set of ad hoc roots (with minimal durability).
    // Then, we need to hide the graph behind the queries such that most queries look only at the proper crate graph, and fall back to ad hoc roots only if there's no results.
    // After this, we should be able to tweak the logic in reload.rs to add newly opened files, which don't belong to any existing crates, to the set of the detached files.
    // //
    /// Project with a set of disjoint files, not belonging to any particular workspace.
    /// Backed by basic sysroot crates for basic completion and highlighting.
    DetachedFiles {
        files: Vec<AbsPathBuf>,
        sysroot: Result<Sysroot, Option<String>>,
        /// Holds cfg flags for the current target. We get those by running
        /// `rustc --print cfg`.
        rustc_cfg: Vec<CfgFlag>,
    },
}

impl fmt::Debug for ProjectWorkspace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Make sure this isn't too verbose.
        match self {
            ProjectWorkspace::Cargo {
                cargo,
                build_scripts: _,
                sysroot,
                rustc,
                rustc_cfg,
                cfg_overrides,
                toolchain,
                target_layout: data_layout,
            } => f
                .debug_struct("Cargo")
                .field("root", &cargo.workspace_root().file_name())
                .field("n_packages", &cargo.packages().len())
                .field("sysroot", &sysroot.is_ok())
                .field(
                    "n_rustc_compiler_crates",
                    &rustc.as_ref().map_or(0, |(rc, _)| rc.packages().len()),
                )
                .field("n_rustc_cfg", &rustc_cfg.len())
                .field("n_cfg_overrides", &cfg_overrides.len())
                .field("toolchain", &toolchain)
                .field("data_layout", &data_layout)
                .finish(),
            ProjectWorkspace::Json { project, sysroot, rustc_cfg, toolchain } => {
                let mut debug_struct = f.debug_struct("Json");
                debug_struct.field("n_crates", &project.n_crates());
                if let Ok(sysroot) = sysroot {
                    debug_struct.field("n_sysroot_crates", &sysroot.crates().len());
                }
                debug_struct.field("toolchain", &toolchain);
                debug_struct.field("n_rustc_cfg", &rustc_cfg.len());
                debug_struct.finish()
            }
            ProjectWorkspace::DetachedFiles { files, sysroot, rustc_cfg } => f
                .debug_struct("DetachedFiles")
                .field("n_files", &files.len())
                .field("sysroot", &sysroot.is_ok())
                .field("n_rustc_cfg", &rustc_cfg.len())
                .finish(),
        }
    }
}

impl ProjectWorkspace {
    pub fn load(
        manifest: ProjectManifest,
        config: &CargoConfig,
        progress: &dyn Fn(String),
    ) -> Result<ProjectWorkspace> {
        let version = |current_dir, cmd_path, prefix: &str| {
            let cargo_version = utf8_stdout({
                let mut cmd = Command::new(cmd_path);
                cmd.envs(&config.extra_env);
                cmd.arg("--version").current_dir(current_dir);
                cmd
            })?;
            anyhow::Ok(
                cargo_version
                    .get(prefix.len()..)
                    .and_then(|it| Version::parse(it.split_whitespace().next()?).ok()),
            )
        };
        let res = match manifest {
            ProjectManifest::ProjectJson(project_json) => {
                let file = fs::read_to_string(&project_json).with_context(|| {
                    format!("Failed to read json file {}", project_json.display())
                })?;
                let data = serde_json::from_str(&file).with_context(|| {
                    format!("Failed to deserialize json file {}", project_json.display())
                })?;
                let project_location = project_json.parent().to_path_buf();
                let toolchain = version(&*project_location, toolchain::rustc(), "rustc ")?;
                let project_json = ProjectJson::new(&project_location, data);
                ProjectWorkspace::load_inline(
                    project_json,
                    config.target.as_deref(),
                    &config.extra_env,
                    toolchain,
                )
            }
            ProjectManifest::CargoToml(cargo_toml) => {
                let toolchain = version(cargo_toml.parent(), toolchain::cargo(), "cargo ")?;
                let meta = CargoWorkspace::fetch_metadata(
                    &cargo_toml,
                    cargo_toml.parent(),
                    config,
                    progress,
                )
                .with_context(|| {
                    format!(
                        "Failed to read Cargo metadata from Cargo.toml file {}, {:?}",
                        cargo_toml.display(),
                        toolchain
                    )
                })?;
                let cargo = CargoWorkspace::new(meta);

                let sysroot = match (&config.sysroot, &config.sysroot_src) {
                    (Some(RustLibSource::Path(path)), None) => {
                        Sysroot::with_sysroot_dir(path.clone()).map_err(|e| {
                          Some(format!("Failed to find sysroot at {}:{e}", path.display()))
                        })
                    }
                    (Some(RustLibSource::Discover), None) => {
                        Sysroot::discover(cargo_toml.parent(), &config.extra_env).map_err(|e| {
                            Some(format!("Failed to find sysroot for Cargo.toml file {}. Is rust-src installed? {e}", cargo_toml.display()))
                        })
                    }
                    (Some(RustLibSource::Path(sysroot)), Some(sysroot_src)) => {
                        Ok(Sysroot::load(sysroot.clone(), sysroot_src.clone()))
                    }
                    (Some(RustLibSource::Discover), Some(sysroot_src)) => {
                        Sysroot::discover_with_src_override(
                            cargo_toml.parent(),
                            &config.extra_env,
                            sysroot_src.clone(),
                        ).map_err(|e| {
                            Some(format!("Failed to find sysroot for Cargo.toml file {}. Is rust-src installed? {e}", cargo_toml.display()))
                        })
                    }
                    (None, _) => Err(None),
                };

                if let Ok(sysroot) = &sysroot {
                    tracing::info!(workspace = %cargo_toml.display(), src_root = %sysroot.src_root().display(), root = %sysroot.root().display(), "Using sysroot");
                }

                let rustc_dir = match &config.rustc_source {
                    Some(RustLibSource::Path(path)) => ManifestPath::try_from(path.clone())
                        .map_err(|p| {
                            Some(format!("rustc source path is not absolute: {}", p.display()))
                        }),
                    Some(RustLibSource::Discover) => {
                        sysroot.as_ref().ok().and_then(Sysroot::discover_rustc).ok_or_else(|| {
                            Some(format!("Failed to discover rustc source for sysroot."))
                        })
                    }
                    None => Err(None),
                };

                let rustc =  rustc_dir.and_then(|rustc_dir| {
                    tracing::info!(workspace = %cargo_toml.display(), rustc_dir = %rustc_dir.display(), "Using rustc source");
                    match CargoWorkspace::fetch_metadata(
                        &rustc_dir,
                        cargo_toml.parent(),
                        &CargoConfig {
                            features: crate::CargoFeatures::default(),
                            ..config.clone()
                        },
                        progress,
                    ) {
                        Ok(meta) => {
                            let workspace = CargoWorkspace::new(meta);
                            let buildscripts = WorkspaceBuildScripts::rustc_crates(
                                &workspace,
                                cargo_toml.parent(),
                                &config.extra_env,
                            );
                            Ok((workspace, buildscripts))
                        }
                        Err(e) => {
                            tracing::error!(
                                %e,
                                "Failed to read Cargo metadata from rustc source at {}",
                                rustc_dir.display()
                            );
                            Err(Some(format!(
                                "Failed to read Cargo metadata from rustc source at {}: {e}",
                                rustc_dir.display())
                            ))
                        }
                    }
                });

                let rustc_cfg =
                    rustc_cfg::get(Some(&cargo_toml), config.target.as_deref(), &config.extra_env);

                let cfg_overrides = config.cfg_overrides.clone();
                let data_layout = target_data_layout::get(
                    Some(&cargo_toml),
                    config.target.as_deref(),
                    &config.extra_env,
                );
                if let Err(e) = &data_layout {
                    tracing::error!(%e, "failed fetching data layout for {cargo_toml:?} workspace");
                }
                ProjectWorkspace::Cargo {
                    cargo,
                    build_scripts: WorkspaceBuildScripts::default(),
                    sysroot,
                    rustc,
                    rustc_cfg,
                    cfg_overrides,
                    toolchain,
                    target_layout: data_layout.map_err(|it| it.to_string()),
                }
            }
        };

        Ok(res)
    }

    pub fn load_inline(
        project_json: ProjectJson,
        target: Option<&str>,
        extra_env: &FxHashMap<String, String>,
        toolchain: Option<Version>,
    ) -> ProjectWorkspace {
        let sysroot = match (project_json.sysroot.clone(), project_json.sysroot_src.clone()) {
            (Some(sysroot), Some(sysroot_src)) => Ok(Sysroot::load(sysroot, sysroot_src)),
            (Some(sysroot), None) => {
                // assume sysroot is structured like rustup's and guess `sysroot_src`
                let sysroot_src =
                    sysroot.join("lib").join("rustlib").join("src").join("rust").join("library");
                Ok(Sysroot::load(sysroot, sysroot_src))
            }
            (None, Some(sysroot_src)) => {
                // assume sysroot is structured like rustup's and guess `sysroot`
                let mut sysroot = sysroot_src.clone();
                for _ in 0..5 {
                    sysroot.pop();
                }
                Ok(Sysroot::load(sysroot, sysroot_src))
            }
            (None, None) => Err(None),
        };
        if let Ok(sysroot) = &sysroot {
            tracing::info!(src_root = %sysroot.src_root().display(), root = %sysroot.root().display(), "Using sysroot");
        }

        let rustc_cfg = rustc_cfg::get(None, target, extra_env);
        ProjectWorkspace::Json { project: project_json, sysroot, rustc_cfg, toolchain }
    }

    pub fn load_detached_files(
        detached_files: Vec<AbsPathBuf>,
        config: &CargoConfig,
    ) -> Result<ProjectWorkspace> {
        let sysroot = match &config.sysroot {
            Some(RustLibSource::Path(path)) => Sysroot::with_sysroot_dir(path.clone())
                .map_err(|e| Some(format!("Failed to find sysroot at {}:{e}", path.display()))),
            Some(RustLibSource::Discover) => {
                let dir = &detached_files
                    .first()
                    .and_then(|it| it.parent())
                    .ok_or_else(|| format_err!("No detached files to load"))?;
                Sysroot::discover(dir, &config.extra_env).map_err(|e| {
                    Some(format!(
                        "Failed to find sysroot for {}. Is rust-src installed? {e}",
                        dir.display()
                    ))
                })
            }
            None => Err(None),
        };
        if let Ok(sysroot) = &sysroot {
            tracing::info!(src_root = %sysroot.src_root().display(), root = %sysroot.root().display(), "Using sysroot");
        }
        let rustc_cfg = rustc_cfg::get(None, None, &Default::default());
        Ok(ProjectWorkspace::DetachedFiles { files: detached_files, sysroot, rustc_cfg })
    }

    /// Runs the build scripts for this [`ProjectWorkspace`].
    pub fn run_build_scripts(
        &self,
        config: &CargoConfig,
        progress: &dyn Fn(String),
    ) -> Result<WorkspaceBuildScripts> {
        match self {
            ProjectWorkspace::Cargo { cargo, toolchain, .. } => {
                WorkspaceBuildScripts::run_for_workspace(config, cargo, progress, toolchain)
                    .with_context(|| {
                        format!(
                            "Failed to run build scripts for {}",
                            &cargo.workspace_root().display()
                        )
                    })
            }
            ProjectWorkspace::Json { .. } | ProjectWorkspace::DetachedFiles { .. } => {
                Ok(WorkspaceBuildScripts::default())
            }
        }
    }

    /// Runs the build scripts for the given [`ProjectWorkspace`]s. Depending on the invocation
    /// strategy this may run a single build process for all project workspaces.
    pub fn run_all_build_scripts(
        workspaces: &[ProjectWorkspace],
        config: &CargoConfig,
        progress: &dyn Fn(String),
    ) -> Vec<Result<WorkspaceBuildScripts>> {
        if matches!(config.invocation_strategy, InvocationStrategy::PerWorkspace)
            || config.run_build_script_command.is_none()
        {
            return workspaces.iter().map(|it| it.run_build_scripts(config, progress)).collect();
        }

        let cargo_ws: Vec<_> = workspaces
            .iter()
            .filter_map(|it| match it {
                ProjectWorkspace::Cargo { cargo, .. } => Some(cargo),
                _ => None,
            })
            .collect();
        let outputs = &mut match WorkspaceBuildScripts::run_once(config, &cargo_ws, progress) {
            Ok(it) => Ok(it.into_iter()),
            // io::Error is not Clone?
            Err(e) => Err(sync::Arc::new(e)),
        };

        workspaces
            .iter()
            .map(|it| match it {
                ProjectWorkspace::Cargo { cargo, .. } => match outputs {
                    Ok(outputs) => Ok(outputs.next().unwrap()),
                    Err(e) => Err(e.clone()).with_context(|| {
                        format!(
                            "Failed to run build scripts for {}",
                            &cargo.workspace_root().display()
                        )
                    }),
                },
                _ => Ok(WorkspaceBuildScripts::default()),
            })
            .collect()
    }

    pub fn set_build_scripts(&mut self, bs: WorkspaceBuildScripts) {
        match self {
            ProjectWorkspace::Cargo { build_scripts, .. } => *build_scripts = bs,
            _ => {
                always!(bs == WorkspaceBuildScripts::default());
            }
        }
    }

    pub fn workspace_definition_path(&self) -> Option<&AbsPath> {
        match self {
            ProjectWorkspace::Cargo { cargo, .. } => Some(cargo.workspace_root()),
            ProjectWorkspace::Json { project, .. } => Some(project.path()),
            ProjectWorkspace::DetachedFiles { .. } => None,
        }
    }

    pub fn find_sysroot_proc_macro_srv(&self) -> Result<AbsPathBuf> {
        match self {
            ProjectWorkspace::Cargo { sysroot: Ok(sysroot), .. }
            | ProjectWorkspace::Json { sysroot: Ok(sysroot), .. }
            | ProjectWorkspace::DetachedFiles { sysroot: Ok(sysroot), .. } => {
                let standalone_server_name =
                    format!("rust-analyzer-proc-macro-srv{}", std::env::consts::EXE_SUFFIX);
                ["libexec", "lib"]
                    .into_iter()
                    .map(|segment| sysroot.root().join(segment).join(&standalone_server_name))
                    .find(|server_path| std::fs::metadata(server_path).is_ok())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "cannot find proc-macro server in sysroot `{}`",
                            sysroot.root().display()
                        )
                    })
            }
            ProjectWorkspace::DetachedFiles { .. } => {
                Err(anyhow::anyhow!("cannot find proc-macro server, no sysroot was found"))
            }
            ProjectWorkspace::Cargo { cargo, .. } => Err(anyhow::anyhow!(
                "cannot find proc-macro-srv, the workspace `{}` is missing a sysroot",
                cargo.workspace_root().display()
            )),
            ProjectWorkspace::Json { project, .. } => Err(anyhow::anyhow!(
                "cannot find proc-macro-srv, the workspace `{}` is missing a sysroot",
                project.path().display()
            )),
        }
    }

    /// Returns the roots for the current `ProjectWorkspace`
    /// The return type contains the path and whether or not
    /// the root is a member of the current workspace
    pub fn to_roots(&self) -> Vec<PackageRoot> {
        let mk_sysroot = |sysroot: Result<&Sysroot, _>, project_root: Option<&AbsPath>| {
            sysroot.map(|sysroot| PackageRoot {
                // mark the sysroot as mutable if it is located inside of the project
                is_local: project_root
                    .map_or(false, |project_root| sysroot.src_root().starts_with(project_root)),
                include: vec![sysroot.src_root().to_path_buf()],
                exclude: Vec::new(),
            })
        };
        match self {
            ProjectWorkspace::Json { project, sysroot, rustc_cfg: _, toolchain: _ } => project
                .crates()
                .map(|(_, krate)| PackageRoot {
                    is_local: krate.is_workspace_member,
                    include: krate.include.clone(),
                    exclude: krate.exclude.clone(),
                })
                .collect::<FxHashSet<_>>()
                .into_iter()
                .chain(mk_sysroot(sysroot.as_ref(), Some(project.path())))
                .collect::<Vec<_>>(),
            ProjectWorkspace::Cargo {
                cargo,
                sysroot,
                rustc,
                rustc_cfg: _,
                cfg_overrides: _,
                build_scripts,
                toolchain: _,
                target_layout: _,
            } => {
                cargo
                    .packages()
                    .map(|pkg| {
                        let is_local = cargo[pkg].is_local;
                        let pkg_root = cargo[pkg].manifest.parent().to_path_buf();

                        let mut include = vec![pkg_root.clone()];
                        let out_dir =
                            build_scripts.get_output(pkg).and_then(|it| it.out_dir.clone());
                        include.extend(out_dir);

                        // In case target's path is manually set in Cargo.toml to be
                        // outside the package root, add its parent as an extra include.
                        // An example of this situation would look like this:
                        //
                        // ```toml
                        // [lib]
                        // path = "../../src/lib.rs"
                        // ```
                        let extra_targets = cargo[pkg]
                            .targets
                            .iter()
                            .filter(|&&tgt| cargo[tgt].kind == TargetKind::Lib)
                            .filter_map(|&tgt| cargo[tgt].root.parent())
                            .map(|tgt| tgt.normalize().to_path_buf())
                            .filter(|path| !path.starts_with(&pkg_root));
                        include.extend(extra_targets);

                        let mut exclude = vec![pkg_root.join(".git")];
                        if is_local {
                            exclude.push(pkg_root.join("target"));
                        } else {
                            exclude.push(pkg_root.join("tests"));
                            exclude.push(pkg_root.join("examples"));
                            exclude.push(pkg_root.join("benches"));
                        }
                        PackageRoot { is_local, include, exclude }
                    })
                    .chain(mk_sysroot(sysroot.as_ref(), Some(cargo.workspace_root())))
                    .chain(rustc.iter().flat_map(|(rustc, _)| {
                        rustc.packages().map(move |krate| PackageRoot {
                            is_local: false,
                            include: vec![rustc[krate].manifest.parent().to_path_buf()],
                            exclude: Vec::new(),
                        })
                    }))
                    .collect()
            }
            ProjectWorkspace::DetachedFiles { files, sysroot, .. } => files
                .iter()
                .map(|detached_file| PackageRoot {
                    is_local: true,
                    include: vec![detached_file.clone()],
                    exclude: Vec::new(),
                })
                .chain(mk_sysroot(sysroot.as_ref(), None))
                .collect(),
        }
    }

    pub fn n_packages(&self) -> usize {
        match self {
            ProjectWorkspace::Json { project, sysroot, .. } => {
                let sysroot_package_len = sysroot.as_ref().map_or(0, |it| it.crates().len());
                sysroot_package_len + project.n_crates()
            }
            ProjectWorkspace::Cargo { cargo, sysroot, rustc, .. } => {
                let rustc_package_len = rustc.as_ref().map_or(0, |(it, _)| it.packages().len());
                let sysroot_package_len = sysroot.as_ref().map_or(0, |it| it.crates().len());
                cargo.packages().len() + sysroot_package_len + rustc_package_len
            }
            ProjectWorkspace::DetachedFiles { sysroot, files, .. } => {
                let sysroot_package_len = sysroot.as_ref().map_or(0, |it| it.crates().len());
                sysroot_package_len + files.len()
            }
        }
    }

    pub fn to_crate_graph(
        &self,
        load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
        extra_env: &FxHashMap<String, String>,
    ) -> (CrateGraph, ProcMacroPaths) {
        let _p = profile::span("ProjectWorkspace::to_crate_graph");

        let (mut crate_graph, proc_macros) = match self {
            ProjectWorkspace::Json { project, sysroot, rustc_cfg, toolchain } => {
                project_json_to_crate_graph(
                    rustc_cfg.clone(),
                    load,
                    project,
                    sysroot.as_ref().ok(),
                    extra_env,
                    Err("rust-project.json projects have no target layout set".into()),
                    toolchain.as_ref().and_then(|it| ReleaseChannel::from_str(it.pre.as_str())),
                )
            }
            ProjectWorkspace::Cargo {
                cargo,
                sysroot,
                rustc,
                rustc_cfg,
                cfg_overrides,
                build_scripts,
                toolchain,
                target_layout,
            } => cargo_to_crate_graph(
                load,
                rustc.as_ref().ok(),
                cargo,
                sysroot.as_ref().ok(),
                rustc_cfg.clone(),
                cfg_overrides,
                None,
                build_scripts,
                match target_layout.as_ref() {
                    Ok(it) => Ok(Arc::from(it.as_str())),
                    Err(it) => Err(Arc::from(it.as_str())),
                },
                toolchain.as_ref().and_then(|it| ReleaseChannel::from_str(it.pre.as_str())),
            ),
            ProjectWorkspace::DetachedFiles { files, sysroot, rustc_cfg } => {
                detached_files_to_crate_graph(
                    rustc_cfg.clone(),
                    load,
                    files,
                    sysroot.as_ref().ok(),
                    Err("detached file projects have no target layout set".into()),
                )
            }
        };
        if crate_graph.patch_cfg_if() {
            tracing::debug!("Patched std to depend on cfg-if")
        } else {
            tracing::debug!("Did not patch std to depend on cfg-if")
        }
        (crate_graph, proc_macros)
    }

    pub fn eq_ignore_build_data(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Cargo {
                    cargo,
                    sysroot,
                    rustc,
                    rustc_cfg,
                    cfg_overrides,
                    toolchain,
                    build_scripts: _,
                    target_layout: _,
                },
                Self::Cargo {
                    cargo: o_cargo,
                    sysroot: o_sysroot,
                    rustc: o_rustc,
                    rustc_cfg: o_rustc_cfg,
                    cfg_overrides: o_cfg_overrides,
                    toolchain: o_toolchain,
                    build_scripts: _,
                    target_layout: _,
                },
            ) => {
                cargo == o_cargo
                    && rustc == o_rustc
                    && rustc_cfg == o_rustc_cfg
                    && cfg_overrides == o_cfg_overrides
                    && toolchain == o_toolchain
                    && sysroot == o_sysroot
            }
            (
                Self::Json { project, sysroot, rustc_cfg, toolchain },
                Self::Json {
                    project: o_project,
                    sysroot: o_sysroot,
                    rustc_cfg: o_rustc_cfg,
                    toolchain: o_toolchain,
                },
            ) => {
                project == o_project
                    && rustc_cfg == o_rustc_cfg
                    && sysroot == o_sysroot
                    && toolchain == o_toolchain
            }
            (
                Self::DetachedFiles { files, sysroot, rustc_cfg },
                Self::DetachedFiles { files: o_files, sysroot: o_sysroot, rustc_cfg: o_rustc_cfg },
            ) => files == o_files && sysroot == o_sysroot && rustc_cfg == o_rustc_cfg,
            _ => false,
        }
    }

    /// Returns `true` if the project workspace is [`Json`].
    ///
    /// [`Json`]: ProjectWorkspace::Json
    #[must_use]
    pub fn is_json(&self) -> bool {
        matches!(self, Self::Json { .. })
    }
}

fn project_json_to_crate_graph(
    rustc_cfg: Vec<CfgFlag>,
    load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
    project: &ProjectJson,
    sysroot: Option<&Sysroot>,
    extra_env: &FxHashMap<String, String>,
    target_layout: TargetLayoutLoadResult,
    channel: Option<ReleaseChannel>,
) -> (CrateGraph, ProcMacroPaths) {
    let mut res = (CrateGraph::default(), ProcMacroPaths::default());
    let (crate_graph, proc_macros) = &mut res;
    let sysroot_deps = sysroot.as_ref().map(|sysroot| {
        sysroot_to_crate_graph(
            crate_graph,
            sysroot,
            rustc_cfg.clone(),
            target_layout.clone(),
            load,
            channel,
        )
    });

    let mut cfg_cache: FxHashMap<&str, Vec<CfgFlag>> = FxHashMap::default();
    let crates: FxHashMap<CrateId, CrateId> = project
        .crates()
        .filter_map(|(crate_id, krate)| Some((crate_id, krate, load(&krate.root_module)?)))
        .map(
            |(
                crate_id,
                Crate {
                    display_name,
                    edition,
                    version,
                    cfg,
                    target,
                    env,
                    proc_macro_dylib_path,
                    is_proc_macro,
                    repository,
                    ..
                },
                file_id,
            )| {
                let env = env.clone().into_iter().collect();

                let target_cfgs = match target.as_deref() {
                    Some(target) => cfg_cache
                        .entry(target)
                        .or_insert_with(|| rustc_cfg::get(None, Some(target), extra_env)),
                    None => &rustc_cfg,
                };

                let crate_graph_crate_id = crate_graph.add_crate_root(
                    file_id,
                    *edition,
                    display_name.clone(),
                    version.clone(),
                    target_cfgs.iter().chain(cfg.iter()).cloned().collect(),
                    None,
                    env,
                    *is_proc_macro,
                    if let Some(name) = display_name.clone() {
                        CrateOrigin::Local {
                            repo: repository.clone(),
                            name: Some(name.canonical_name().to_string()),
                        }
                    } else {
                        CrateOrigin::Local { repo: None, name: None }
                    },
                    target_layout.clone(),
                    channel,
                );
                if *is_proc_macro {
                    if let Some(path) = proc_macro_dylib_path.clone() {
                        let node = Ok((
                            display_name.as_ref().map(|it| it.canonical_name().to_owned()),
                            path,
                        ));
                        proc_macros.insert(crate_graph_crate_id, node);
                    }
                }
                (crate_id, crate_graph_crate_id)
            },
        )
        .collect();

    for (from, krate) in project.crates() {
        if let Some(&from) = crates.get(&from) {
            if let Some((public_deps, libproc_macro)) = &sysroot_deps {
                public_deps.add_to_crate_graph(crate_graph, from);
                if let Some(proc_macro) = libproc_macro {
                    add_proc_macro_dep(crate_graph, from, *proc_macro, krate.is_proc_macro);
                }
            }

            for dep in &krate.deps {
                if let Some(&to) = crates.get(&dep.crate_id) {
                    add_dep(crate_graph, from, dep.name.clone(), to)
                }
            }
        }
    }
    res
}

fn cargo_to_crate_graph(
    load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
    rustc: Option<&(CargoWorkspace, WorkspaceBuildScripts)>,
    cargo: &CargoWorkspace,
    sysroot: Option<&Sysroot>,
    rustc_cfg: Vec<CfgFlag>,
    override_cfg: &CfgOverrides,
    // Don't compute cfg and use this if present
    forced_cfg: Option<CfgOptions>,
    build_scripts: &WorkspaceBuildScripts,
    target_layout: TargetLayoutLoadResult,
    channel: Option<ReleaseChannel>,
) -> (CrateGraph, ProcMacroPaths) {
    let _p = profile::span("cargo_to_crate_graph");
    let mut res = (CrateGraph::default(), ProcMacroPaths::default());
    let crate_graph = &mut res.0;
    let proc_macros = &mut res.1;
    let (public_deps, libproc_macro) = match sysroot {
        Some(sysroot) => sysroot_to_crate_graph(
            crate_graph,
            sysroot,
            rustc_cfg.clone(),
            target_layout.clone(),
            load,
            channel,
        ),
        None => (SysrootPublicDeps::default(), None),
    };

    let cfg_options = {
        let mut cfg_options = CfgOptions::default();
        cfg_options.extend(rustc_cfg);
        cfg_options.insert_atom("debug_assertions".into());
        cfg_options
    };

    // Mapping of a package to its library target
    let mut pkg_to_lib_crate = FxHashMap::default();
    let mut pkg_crates = FxHashMap::default();
    // Does any crate signal to rust-analyzer that they need the rustc_private crates?
    let mut has_private = false;

    // Next, create crates for each package, target pair
    for pkg in cargo.packages() {
        has_private |= cargo[pkg].metadata.rustc_private;

        let cfg_options = forced_cfg.clone().unwrap_or_else(|| {
            let mut cfg_options = cfg_options.clone();

            // Add test cfg for local crates
            if cargo[pkg].is_local {
                cfg_options.insert_atom("test".into());
            }

            if !override_cfg.global.is_empty() {
                cfg_options.apply_diff(override_cfg.global.clone());
            };
            if let Some(diff) = override_cfg.selective.get(&cargo[pkg].name) {
                // FIXME: this is sort of a hack to deal with #![cfg(not(test))] vanishing such as seen
                // in ed25519_dalek (#7243), and libcore (#9203) (although you only hit that one while
                // working on rust-lang/rust as that's the only time it appears outside sysroot).
                //
                // A more ideal solution might be to reanalyze crates based on where the cursor is and
                // figure out the set of cfgs that would have to apply to make it active.

                cfg_options.apply_diff(diff.clone());
            };
            cfg_options
        });

        let mut lib_tgt = None;
        for &tgt in cargo[pkg].targets.iter() {
            if cargo[tgt].kind != TargetKind::Lib && !cargo[pkg].is_member {
                // For non-workspace-members, Cargo does not resolve dev-dependencies, so we don't
                // add any targets except the library target, since those will not work correctly if
                // they use dev-dependencies.
                // In fact, they can break quite badly if multiple client workspaces get merged:
                // https://github.com/rust-lang/rust-analyzer/issues/11300
                continue;
            }
            let &TargetData { ref name, kind, is_proc_macro, ref root, .. } = &cargo[tgt];

            if kind == TargetKind::Lib
                && sysroot.map_or(false, |sysroot| root.starts_with(sysroot.src_root()))
            {
                if let Some(&(_, crate_id, _)) =
                    public_deps.deps.iter().find(|(dep_name, ..)| dep_name.as_smol_str() == name)
                {
                    pkg_crates.entry(pkg).or_insert_with(Vec::new).push((crate_id, kind));

                    lib_tgt = Some((crate_id, name.clone()));
                    pkg_to_lib_crate.insert(pkg, crate_id);
                    // sysroot is inside the workspace, prevent the sysroot crates from being duplicated here
                    continue;
                }
            }

            let Some(file_id) = load(root) else { continue };

            let crate_id = add_target_crate_root(
                crate_graph,
                proc_macros,
                &cargo[pkg],
                build_scripts.get_output(pkg),
                cfg_options.clone(),
                file_id,
                name,
                is_proc_macro,
                target_layout.clone(),
                false,
                channel,
            );
            if kind == TargetKind::Lib {
                lib_tgt = Some((crate_id, name.clone()));
                pkg_to_lib_crate.insert(pkg, crate_id);
            }
            // Even crates that don't set proc-macro = true are allowed to depend on proc_macro
            // (just none of the APIs work when called outside of a proc macro).
            if let Some(proc_macro) = libproc_macro {
                add_proc_macro_dep(crate_graph, crate_id, proc_macro, is_proc_macro);
            }

            pkg_crates.entry(pkg).or_insert_with(Vec::new).push((crate_id, kind));
        }

        // Set deps to the core, std and to the lib target of the current package
        for &(from, kind) in pkg_crates.get(&pkg).into_iter().flatten() {
            // Add sysroot deps first so that a lib target named `core` etc. can overwrite them.
            public_deps.add_to_crate_graph(crate_graph, from);

            // Add dep edge of all targets to the package's lib target
            if let Some((to, name)) = lib_tgt.clone() {
                if to != from && kind != TargetKind::BuildScript {
                    // (build script can not depend on its library target)

                    // For root projects with dashes in their name,
                    // cargo metadata does not do any normalization,
                    // so we do it ourselves currently
                    let name = CrateName::normalize_dashes(&name);
                    add_dep(crate_graph, from, name, to);
                }
            }
        }
    }

    // Now add a dep edge from all targets of upstream to the lib
    // target of downstream.
    for pkg in cargo.packages() {
        for dep in &cargo[pkg].dependencies {
            let Some(&to) = pkg_to_lib_crate.get(&dep.pkg) else { continue };
            let Some(targets) = pkg_crates.get(&pkg) else { continue };

            let name = CrateName::new(&dep.name).unwrap();
            for &(from, kind) in targets {
                // Build scripts may only depend on build dependencies.
                if (dep.kind == DepKind::Build) != (kind == TargetKind::BuildScript) {
                    continue;
                }

                add_dep(crate_graph, from, name.clone(), to)
            }
        }
    }

    if has_private {
        // If the user provided a path to rustc sources, we add all the rustc_private crates
        // and create dependencies on them for the crates which opt-in to that
        if let Some((rustc_workspace, rustc_build_scripts)) = rustc {
            handle_rustc_crates(
                crate_graph,
                proc_macros,
                &mut pkg_to_lib_crate,
                load,
                rustc_workspace,
                cargo,
                &public_deps,
                libproc_macro,
                &pkg_crates,
                &cfg_options,
                override_cfg,
                if rustc_workspace.workspace_root() == cargo.workspace_root() {
                    // the rustc workspace does not use the installed toolchain's proc-macro server
                    // so we need to make sure we don't use the pre compiled proc-macros there either
                    build_scripts
                } else {
                    rustc_build_scripts
                },
                target_layout,
                channel,
            );
        }
    }
    res
}

fn detached_files_to_crate_graph(
    rustc_cfg: Vec<CfgFlag>,
    load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
    detached_files: &[AbsPathBuf],
    sysroot: Option<&Sysroot>,
    target_layout: TargetLayoutLoadResult,
) -> (CrateGraph, ProcMacroPaths) {
    let _p = profile::span("detached_files_to_crate_graph");
    let mut crate_graph = CrateGraph::default();
    let (public_deps, _libproc_macro) = match sysroot {
        Some(sysroot) => sysroot_to_crate_graph(
            &mut crate_graph,
            sysroot,
            rustc_cfg.clone(),
            target_layout.clone(),
            load,
            None,
        ),
        None => (SysrootPublicDeps::default(), None),
    };

    let mut cfg_options = CfgOptions::default();
    cfg_options.extend(rustc_cfg);

    for detached_file in detached_files {
        let file_id = match load(detached_file) {
            Some(file_id) => file_id,
            None => {
                tracing::error!("Failed to load detached file {:?}", detached_file);
                continue;
            }
        };
        let display_name = detached_file
            .file_stem()
            .and_then(|os_str| os_str.to_str())
            .map(|file_stem| CrateDisplayName::from_canonical_name(file_stem.to_string()));
        let detached_file_crate = crate_graph.add_crate_root(
            file_id,
            Edition::CURRENT,
            display_name.clone(),
            None,
            cfg_options.clone(),
            None,
            Env::default(),
            false,
            CrateOrigin::Local {
                repo: None,
                name: display_name.map(|n| n.canonical_name().to_string()),
            },
            target_layout.clone(),
            None,
        );

        public_deps.add_to_crate_graph(&mut crate_graph, detached_file_crate);
    }
    (crate_graph, FxHashMap::default())
}

fn handle_rustc_crates(
    crate_graph: &mut CrateGraph,
    proc_macros: &mut ProcMacroPaths,
    pkg_to_lib_crate: &mut FxHashMap<Package, CrateId>,
    load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
    rustc_workspace: &CargoWorkspace,
    cargo: &CargoWorkspace,
    public_deps: &SysrootPublicDeps,
    libproc_macro: Option<CrateId>,
    pkg_crates: &FxHashMap<Package, Vec<(CrateId, TargetKind)>>,
    cfg_options: &CfgOptions,
    override_cfg: &CfgOverrides,
    build_scripts: &WorkspaceBuildScripts,
    target_layout: TargetLayoutLoadResult,
    channel: Option<ReleaseChannel>,
) {
    let mut rustc_pkg_crates = FxHashMap::default();
    // The root package of the rustc-dev component is rustc_driver, so we match that
    let root_pkg =
        rustc_workspace.packages().find(|&package| rustc_workspace[package].name == "rustc_driver");
    // The rustc workspace might be incomplete (such as if rustc-dev is not
    // installed for the current toolchain) and `rustc_source` is set to discover.
    if let Some(root_pkg) = root_pkg {
        // Iterate through every crate in the dependency subtree of rustc_driver using BFS
        let mut queue = VecDeque::new();
        queue.push_back(root_pkg);
        while let Some(pkg) = queue.pop_front() {
            // Don't duplicate packages if they are dependent on a diamond pattern
            // N.B. if this line is omitted, we try to analyze over 4_800_000 crates
            // which is not ideal
            if rustc_pkg_crates.contains_key(&pkg) {
                continue;
            }
            for dep in &rustc_workspace[pkg].dependencies {
                queue.push_back(dep.pkg);
            }

            let mut cfg_options = cfg_options.clone();

            if !override_cfg.global.is_empty() {
                cfg_options.apply_diff(override_cfg.global.clone());
            };
            if let Some(diff) = override_cfg.selective.get(&rustc_workspace[pkg].name) {
                // FIXME: this is sort of a hack to deal with #![cfg(not(test))] vanishing such as seen
                // in ed25519_dalek (#7243), and libcore (#9203) (although you only hit that one while
                // working on rust-lang/rust as that's the only time it appears outside sysroot).
                //
                // A more ideal solution might be to reanalyze crates based on where the cursor is and
                // figure out the set of cfgs that would have to apply to make it active.

                cfg_options.apply_diff(diff.clone());
            };

            for &tgt in rustc_workspace[pkg].targets.iter() {
                if rustc_workspace[tgt].kind != TargetKind::Lib {
                    continue;
                }
                if let Some(file_id) = load(&rustc_workspace[tgt].root) {
                    let crate_id = add_target_crate_root(
                        crate_graph,
                        proc_macros,
                        &rustc_workspace[pkg],
                        build_scripts.get_output(pkg),
                        cfg_options.clone(),
                        file_id,
                        &rustc_workspace[tgt].name,
                        rustc_workspace[tgt].is_proc_macro,
                        target_layout.clone(),
                        true,
                        channel,
                    );
                    pkg_to_lib_crate.insert(pkg, crate_id);
                    // Add dependencies on core / std / alloc for this crate
                    public_deps.add_to_crate_graph(crate_graph, crate_id);
                    if let Some(proc_macro) = libproc_macro {
                        add_proc_macro_dep(
                            crate_graph,
                            crate_id,
                            proc_macro,
                            rustc_workspace[tgt].is_proc_macro,
                        );
                    }
                    rustc_pkg_crates.entry(pkg).or_insert_with(Vec::new).push(crate_id);
                }
            }
        }
    }
    // Now add a dep edge from all targets of upstream to the lib
    // target of downstream.
    for pkg in rustc_pkg_crates.keys().copied() {
        for dep in rustc_workspace[pkg].dependencies.iter() {
            let name = CrateName::new(&dep.name).unwrap();
            if let Some(&to) = pkg_to_lib_crate.get(&dep.pkg) {
                for &from in rustc_pkg_crates.get(&pkg).into_iter().flatten() {
                    add_dep(crate_graph, from, name.clone(), to);
                }
            }
        }
    }
    // Add a dependency on the rustc_private crates for all targets of each package
    // which opts in
    for dep in rustc_workspace.packages() {
        let name = CrateName::normalize_dashes(&rustc_workspace[dep].name);

        if let Some(&to) = pkg_to_lib_crate.get(&dep) {
            for pkg in cargo.packages() {
                let package = &cargo[pkg];
                if !package.metadata.rustc_private {
                    continue;
                }
                for (from, _) in pkg_crates.get(&pkg).into_iter().flatten() {
                    // Avoid creating duplicate dependencies
                    // This avoids the situation where `from` depends on e.g. `arrayvec`, but
                    // `rust_analyzer` thinks that it should use the one from the `rustc_source`
                    // instead of the one from `crates.io`
                    if !crate_graph[*from].dependencies.iter().any(|d| d.name == name) {
                        add_dep(crate_graph, *from, name.clone(), to);
                    }
                }
            }
        }
    }
}

fn add_target_crate_root(
    crate_graph: &mut CrateGraph,
    proc_macros: &mut ProcMacroPaths,
    pkg: &PackageData,
    build_data: Option<&BuildScriptOutput>,
    cfg_options: CfgOptions,
    file_id: FileId,
    cargo_name: &str,
    is_proc_macro: bool,
    target_layout: TargetLayoutLoadResult,
    rustc_crate: bool,
    channel: Option<ReleaseChannel>,
) -> CrateId {
    let edition = pkg.edition;
    let potential_cfg_options = if pkg.features.is_empty() {
        None
    } else {
        let mut potential_cfg_options = cfg_options.clone();
        potential_cfg_options.extend(
            pkg.features
                .iter()
                .map(|feat| CfgFlag::KeyValue { key: "feature".into(), value: feat.0.into() }),
        );
        Some(potential_cfg_options)
    };
    let cfg_options = {
        let mut opts = cfg_options;
        for feature in pkg.active_features.iter() {
            opts.insert_key_value("feature".into(), feature.into());
        }
        if let Some(cfgs) = build_data.as_ref().map(|it| &it.cfgs) {
            opts.extend(cfgs.iter().cloned());
        }
        opts
    };

    let mut env = Env::default();
    inject_cargo_env(pkg, &mut env);

    if let Some(envs) = build_data.map(|it| &it.envs) {
        for (k, v) in envs {
            env.set(k, v.clone());
        }
    }

    let display_name = CrateDisplayName::from_canonical_name(cargo_name.to_string());
    let crate_id = crate_graph.add_crate_root(
        file_id,
        edition,
        Some(display_name),
        Some(pkg.version.to_string()),
        cfg_options,
        potential_cfg_options,
        env,
        is_proc_macro,
        if rustc_crate {
            CrateOrigin::Rustc { name: pkg.name.clone() }
        } else if pkg.is_member {
            CrateOrigin::Local { repo: pkg.repository.clone(), name: Some(pkg.name.clone()) }
        } else {
            CrateOrigin::Library { repo: pkg.repository.clone(), name: pkg.name.clone() }
        },
        target_layout,
        channel,
    );
    if is_proc_macro {
        let proc_macro = match build_data.as_ref().map(|it| it.proc_macro_dylib_path.as_ref()) {
            Some(it) => it.cloned().map(|path| Ok((Some(cargo_name.to_owned()), path))),
            None => Some(Err("crate has not yet been built".to_owned())),
        };
        if let Some(proc_macro) = proc_macro {
            proc_macros.insert(crate_id, proc_macro);
        }
    }

    crate_id
}

#[derive(Default)]
struct SysrootPublicDeps {
    deps: Vec<(CrateName, CrateId, bool)>,
}

impl SysrootPublicDeps {
    /// Makes `from` depend on the public sysroot crates.
    fn add_to_crate_graph(&self, crate_graph: &mut CrateGraph, from: CrateId) {
        for (name, krate, prelude) in &self.deps {
            add_dep_with_prelude(crate_graph, from, name.clone(), *krate, *prelude);
        }
    }
}

fn sysroot_to_crate_graph(
    crate_graph: &mut CrateGraph,
    sysroot: &Sysroot,
    rustc_cfg: Vec<CfgFlag>,
    target_layout: TargetLayoutLoadResult,
    load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
    channel: Option<ReleaseChannel>,
) -> (SysrootPublicDeps, Option<CrateId>) {
    let _p = profile::span("sysroot_to_crate_graph");
    let mut cfg_options = CfgOptions::default();
    cfg_options.extend(rustc_cfg.clone());
    let sysroot_crates: FxHashMap<SysrootCrate, CrateId> = match &sysroot.hack_cargo_workspace {
        Some(cargo) => handle_hack_cargo_workspace(
            load,
            cargo,
            rustc_cfg,
            cfg_options,
            target_layout,
            channel,
            crate_graph,
            sysroot,
        ),
        None => sysroot
            .crates()
            .filter_map(|krate| {
                let file_id = load(&sysroot[krate].root)?;

                let env = Env::default();
                let display_name =
                    CrateDisplayName::from_canonical_name(sysroot[krate].name.clone());
                let crate_id = crate_graph.add_crate_root(
                    file_id,
                    Edition::CURRENT,
                    Some(display_name),
                    None,
                    cfg_options.clone(),
                    None,
                    env,
                    false,
                    CrateOrigin::Lang(LangCrateOrigin::from(&*sysroot[krate].name)),
                    target_layout.clone(),
                    channel,
                );
                Some((krate, crate_id))
            })
            .collect(),
    };
    for from in sysroot.crates() {
        for &to in sysroot[from].deps.iter() {
            let name = CrateName::new(&sysroot[to].name).unwrap();
            if let (Some(&from), Some(&to)) = (sysroot_crates.get(&from), sysroot_crates.get(&to)) {
                add_dep(crate_graph, from, name, to);
            }
        }
    }

    let public_deps = SysrootPublicDeps {
        deps: sysroot
            .public_deps()
            .map(|(name, idx, prelude)| (name, sysroot_crates[&idx], prelude))
            .collect::<Vec<_>>(),
    };

    let libproc_macro = sysroot.proc_macro().and_then(|it| sysroot_crates.get(&it).copied());
    (public_deps, libproc_macro)
}

fn handle_hack_cargo_workspace(
    load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
    cargo: &CargoWorkspace,
    rustc_cfg: Vec<CfgFlag>,
    cfg_options: CfgOptions,
    target_layout: Result<Arc<str>, Arc<str>>,
    channel: Option<ReleaseChannel>,
    crate_graph: &mut CrateGraph,
    sysroot: &Sysroot,
) -> FxHashMap<SysrootCrate, CrateId> {
    let (cg, mut pm) = cargo_to_crate_graph(
        load,
        None,
        cargo,
        None,
        rustc_cfg,
        &CfgOverrides::default(),
        Some(cfg_options),
        &WorkspaceBuildScripts::default(),
        target_layout,
        channel,
    );
    crate_graph.extend(cg, &mut pm);
    for crate_name in ["std", "alloc", "core"] {
        let original = crate_graph
            .iter()
            .find(|x| {
                crate_graph[*x]
                    .display_name
                    .as_ref()
                    .map(|x| x.canonical_name() == crate_name)
                    .unwrap_or(false)
            })
            .unwrap();
        let fake_crate_name = format!("rustc-std-workspace-{}", crate_name);
        let fake = crate_graph
            .iter()
            .find(|x| {
                crate_graph[*x]
                    .display_name
                    .as_ref()
                    .map(|x| x.canonical_name() == fake_crate_name)
                    .unwrap_or(false)
            })
            .unwrap();
        crate_graph.remove_and_replace(fake, original).unwrap();
    }
    for (_, c) in crate_graph.iter_mut() {
        if c.origin.is_local() {
            // LangCrateOrigin::Other is good enough for a hack.
            c.origin = CrateOrigin::Lang(LangCrateOrigin::Other);
        }
    }
    sysroot
        .crates()
        .filter_map(|krate| {
            let file_id = load(&sysroot[krate].root)?;
            let crate_id = crate_graph.crate_id_for_crate_root(file_id)?;
            Some((krate, crate_id))
        })
        .collect()
}

fn add_dep(graph: &mut CrateGraph, from: CrateId, name: CrateName, to: CrateId) {
    add_dep_inner(graph, from, Dependency::new(name, to))
}

fn add_dep_with_prelude(
    graph: &mut CrateGraph,
    from: CrateId,
    name: CrateName,
    to: CrateId,
    prelude: bool,
) {
    add_dep_inner(graph, from, Dependency::with_prelude(name, to, prelude))
}

fn add_proc_macro_dep(crate_graph: &mut CrateGraph, from: CrateId, to: CrateId, prelude: bool) {
    add_dep_with_prelude(crate_graph, from, CrateName::new("proc_macro").unwrap(), to, prelude);
}

fn add_dep_inner(graph: &mut CrateGraph, from: CrateId, dep: Dependency) {
    if let Err(err) = graph.add_dep(from, dep) {
        tracing::error!("{}", err)
    }
}

/// Recreates the compile-time environment variables that Cargo sets.
///
/// Should be synced with
/// <https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-crates>
///
/// FIXME: ask Cargo to provide this data instead of re-deriving.
fn inject_cargo_env(package: &PackageData, env: &mut Env) {
    // FIXME: Missing variables:
    // CARGO_BIN_NAME, CARGO_BIN_EXE_<name>

    let manifest_dir = package.manifest.parent();
    env.set("CARGO_MANIFEST_DIR", manifest_dir.as_os_str().to_string_lossy().into_owned());

    // Not always right, but works for common cases.
    env.set("CARGO", "cargo".into());

    env.set("CARGO_PKG_VERSION", package.version.to_string());
    env.set("CARGO_PKG_VERSION_MAJOR", package.version.major.to_string());
    env.set("CARGO_PKG_VERSION_MINOR", package.version.minor.to_string());
    env.set("CARGO_PKG_VERSION_PATCH", package.version.patch.to_string());
    env.set("CARGO_PKG_VERSION_PRE", package.version.pre.to_string());

    env.set("CARGO_PKG_AUTHORS", String::new());

    env.set("CARGO_PKG_NAME", package.name.clone());
    // FIXME: This isn't really correct (a package can have many crates with different names), but
    // it's better than leaving the variable unset.
    env.set("CARGO_CRATE_NAME", CrateName::normalize_dashes(&package.name).to_string());
    env.set("CARGO_PKG_DESCRIPTION", String::new());
    env.set("CARGO_PKG_HOMEPAGE", String::new());
    env.set("CARGO_PKG_REPOSITORY", String::new());
    env.set("CARGO_PKG_LICENSE", String::new());

    env.set("CARGO_PKG_LICENSE_FILE", String::new());
}
