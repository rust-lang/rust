//! Handles lowering of build-system specific workspace information (`cargo
//! metadata` or `rust-project.json`) into representation stored in the salsa
//! database -- `CrateGraph`.

use std::{collections::VecDeque, fmt, fs, process::Command};

use anyhow::{format_err, Context, Result};
use base_db::{
    CrateDisplayName, CrateGraph, CrateId, CrateName, CrateOrigin, Dependency, Edition, Env,
    FileId, LangCrateOrigin, ProcMacroLoadResult,
};
use cfg::{CfgDiff, CfgOptions};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::{FxHashMap, FxHashSet};
use semver::Version;
use stdx::always;

use crate::{
    build_scripts::BuildScriptOutput,
    cargo_workspace::{DepKind, PackageData, RustcSource},
    cfg_flag::CfgFlag,
    rustc_cfg,
    sysroot::SysrootCrate,
    utf8_stdout, CargoConfig, CargoWorkspace, ManifestPath, ProjectJson, ProjectManifest, Sysroot,
    TargetKind, WorkspaceBuildScripts,
};

/// A set of cfg-overrides per crate.
///
/// `Wildcard(..)` is useful e.g. disabling `#[cfg(test)]` on all crates,
/// without having to first obtain a list of all crates.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CfgOverrides {
    /// A single global set of overrides matching all crates.
    Wildcard(CfgDiff),
    /// A set of overrides matching specific crates.
    Selective(FxHashMap<String, CfgDiff>),
}

impl Default for CfgOverrides {
    fn default() -> Self {
        Self::Selective(FxHashMap::default())
    }
}

impl CfgOverrides {
    pub fn len(&self) -> usize {
        match self {
            CfgOverrides::Wildcard(_) => 1,
            CfgOverrides::Selective(hash_map) => hash_map.len(),
        }
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

#[derive(Clone, Eq, PartialEq)]
pub enum ProjectWorkspace {
    /// Project workspace was discovered by running `cargo metadata` and `rustc --print sysroot`.
    Cargo {
        cargo: CargoWorkspace,
        build_scripts: WorkspaceBuildScripts,
        sysroot: Option<Sysroot>,
        rustc: Option<CargoWorkspace>,
        /// Holds cfg flags for the current target. We get those by running
        /// `rustc --print cfg`.
        ///
        /// FIXME: make this a per-crate map, as, eg, build.rs might have a
        /// different target.
        rustc_cfg: Vec<CfgFlag>,
        cfg_overrides: CfgOverrides,
        toolchain: Option<Version>,
    },
    /// Project workspace was manually specified using a `rust-project.json` file.
    Json { project: ProjectJson, sysroot: Option<Sysroot>, rustc_cfg: Vec<CfgFlag> },

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
    DetachedFiles { files: Vec<AbsPathBuf>, sysroot: Sysroot, rustc_cfg: Vec<CfgFlag> },
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
            } => f
                .debug_struct("Cargo")
                .field("root", &cargo.workspace_root().file_name())
                .field("n_packages", &cargo.packages().len())
                .field("sysroot", &sysroot.is_some())
                .field(
                    "n_rustc_compiler_crates",
                    &rustc.as_ref().map_or(0, |rc| rc.packages().len()),
                )
                .field("n_rustc_cfg", &rustc_cfg.len())
                .field("n_cfg_overrides", &cfg_overrides.len())
                .field("toolchain", &toolchain)
                .finish(),
            ProjectWorkspace::Json { project, sysroot, rustc_cfg } => {
                let mut debug_struct = f.debug_struct("Json");
                debug_struct.field("n_crates", &project.n_crates());
                if let Some(sysroot) = sysroot {
                    debug_struct.field("n_sysroot_crates", &sysroot.crates().len());
                }
                debug_struct.field("n_rustc_cfg", &rustc_cfg.len());
                debug_struct.finish()
            }
            ProjectWorkspace::DetachedFiles { files, sysroot, rustc_cfg } => f
                .debug_struct("DetachedFiles")
                .field("n_files", &files.len())
                .field("n_sysroot_crates", &sysroot.crates().len())
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
        let res = match manifest {
            ProjectManifest::ProjectJson(project_json) => {
                let file = fs::read_to_string(&project_json).with_context(|| {
                    format!("Failed to read json file {}", project_json.display())
                })?;
                let data = serde_json::from_str(&file).with_context(|| {
                    format!("Failed to deserialize json file {}", project_json.display())
                })?;
                let project_location = project_json.parent().to_path_buf();
                let project_json = ProjectJson::new(&project_location, data);
                ProjectWorkspace::load_inline(project_json, config.target.as_deref())?
            }
            ProjectManifest::CargoToml(cargo_toml) => {
                let cargo_version = utf8_stdout({
                    let mut cmd = Command::new(toolchain::cargo());
                    cmd.arg("--version");
                    cmd
                })?;
                let toolchain = cargo_version
                    .get("cargo ".len()..)
                    .and_then(|it| Version::parse(it.split_whitespace().next()?).ok());

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

                let sysroot = if config.no_sysroot {
                    None
                } else {
                    Some(Sysroot::discover(cargo_toml.parent()).with_context(|| {
                        format!(
                            "Failed to find sysroot for Cargo.toml file {}. Is rust-src installed?",
                            cargo_toml.display()
                        )
                    })?)
                };

                let rustc_dir = match &config.rustc_source {
                    Some(RustcSource::Path(path)) => ManifestPath::try_from(path.clone()).ok(),
                    Some(RustcSource::Discover) => Sysroot::discover_rustc(&cargo_toml),
                    None => None,
                };

                let rustc = match rustc_dir {
                    Some(rustc_dir) => Some({
                        let meta = CargoWorkspace::fetch_metadata(
                            &rustc_dir,
                            cargo_toml.parent(),
                            config,
                            progress,
                        )
                        .with_context(|| {
                            "Failed to read Cargo metadata for Rust sources".to_string()
                        })?;
                        CargoWorkspace::new(meta)
                    }),
                    None => None,
                };

                let rustc_cfg = rustc_cfg::get(Some(&cargo_toml), config.target.as_deref());

                let cfg_overrides = config.cfg_overrides();
                ProjectWorkspace::Cargo {
                    cargo,
                    build_scripts: WorkspaceBuildScripts::default(),
                    sysroot,
                    rustc,
                    rustc_cfg,
                    cfg_overrides,
                    toolchain,
                }
            }
        };

        Ok(res)
    }

    pub fn load_inline(
        project_json: ProjectJson,
        target: Option<&str>,
    ) -> Result<ProjectWorkspace> {
        let sysroot = match (project_json.sysroot.clone(), project_json.sysroot_src.clone()) {
            (Some(sysroot), Some(sysroot_src)) => Some(Sysroot::load(sysroot, sysroot_src)?),
            (Some(sysroot), None) => {
                // assume sysroot is structured like rustup's and guess `sysroot_src`
                let sysroot_src =
                    sysroot.join("lib").join("rustlib").join("src").join("rust").join("library");

                Some(Sysroot::load(sysroot, sysroot_src)?)
            }
            (None, Some(sysroot_src)) => {
                // assume sysroot is structured like rustup's and guess `sysroot`
                let mut sysroot = sysroot_src.clone();
                for _ in 0..5 {
                    sysroot.pop();
                }
                Some(Sysroot::load(sysroot, sysroot_src)?)
            }
            (None, None) => None,
        };

        let rustc_cfg = rustc_cfg::get(None, target);
        Ok(ProjectWorkspace::Json { project: project_json, sysroot, rustc_cfg })
    }

    pub fn load_detached_files(detached_files: Vec<AbsPathBuf>) -> Result<ProjectWorkspace> {
        let sysroot = Sysroot::discover(
            detached_files
                .first()
                .and_then(|it| it.parent())
                .ok_or_else(|| format_err!("No detached files to load"))?,
        )?;
        let rustc_cfg = rustc_cfg::get(None, None);
        Ok(ProjectWorkspace::DetachedFiles { files: detached_files, sysroot, rustc_cfg })
    }

    pub fn run_build_scripts(
        &self,
        config: &CargoConfig,
        progress: &dyn Fn(String),
    ) -> Result<WorkspaceBuildScripts> {
        match self {
            ProjectWorkspace::Cargo { cargo, toolchain, .. } => {
                WorkspaceBuildScripts::run(config, cargo, progress, toolchain).with_context(|| {
                    format!("Failed to run build scripts for {}", &cargo.workspace_root().display())
                })
            }
            ProjectWorkspace::Json { .. } | ProjectWorkspace::DetachedFiles { .. } => {
                Ok(WorkspaceBuildScripts::default())
            }
        }
    }

    pub fn set_build_scripts(&mut self, bs: WorkspaceBuildScripts) {
        match self {
            ProjectWorkspace::Cargo { build_scripts, .. } => *build_scripts = bs,
            _ => {
                always!(bs == WorkspaceBuildScripts::default());
            }
        }
    }

    /// Returns the roots for the current `ProjectWorkspace`
    /// The return type contains the path and whether or not
    /// the root is a member of the current workspace
    pub fn to_roots(&self) -> Vec<PackageRoot> {
        match self {
            ProjectWorkspace::Json { project, sysroot, rustc_cfg: _ } => project
                .crates()
                .map(|(_, krate)| PackageRoot {
                    is_local: krate.is_workspace_member,
                    include: krate.include.clone(),
                    exclude: krate.exclude.clone(),
                })
                .collect::<FxHashSet<_>>()
                .into_iter()
                .chain(sysroot.as_ref().into_iter().flat_map(|sysroot| {
                    sysroot.crates().map(move |krate| PackageRoot {
                        is_local: false,
                        include: vec![sysroot[krate].root.parent().to_path_buf()],
                        exclude: Vec::new(),
                    })
                }))
                .collect::<Vec<_>>(),
            ProjectWorkspace::Cargo {
                cargo,
                sysroot,
                rustc,
                rustc_cfg: _,
                cfg_overrides: _,
                build_scripts,
                toolchain: _,
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
                    .chain(sysroot.iter().map(|sysroot| PackageRoot {
                        is_local: false,
                        include: vec![sysroot.src_root().to_path_buf()],
                        exclude: Vec::new(),
                    }))
                    .chain(rustc.iter().flat_map(|rustc| {
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
                .chain(sysroot.crates().map(|krate| PackageRoot {
                    is_local: false,
                    include: vec![sysroot[krate].root.parent().to_path_buf()],
                    exclude: Vec::new(),
                }))
                .collect(),
        }
    }

    pub fn n_packages(&self) -> usize {
        match self {
            ProjectWorkspace::Json { project, .. } => project.n_crates(),
            ProjectWorkspace::Cargo { cargo, sysroot, rustc, .. } => {
                let rustc_package_len = rustc.as_ref().map_or(0, |it| it.packages().len());
                let sysroot_package_len = sysroot.as_ref().map_or(0, |it| it.crates().len());
                cargo.packages().len() + sysroot_package_len + rustc_package_len
            }
            ProjectWorkspace::DetachedFiles { sysroot, files, .. } => {
                sysroot.crates().len() + files.len()
            }
        }
    }

    pub fn to_crate_graph(
        &self,
        load_proc_macro: &mut dyn FnMut(&str, &AbsPath) -> ProcMacroLoadResult,
        load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
    ) -> CrateGraph {
        let _p = profile::span("ProjectWorkspace::to_crate_graph");

        let mut crate_graph = match self {
            ProjectWorkspace::Json { project, sysroot, rustc_cfg } => project_json_to_crate_graph(
                rustc_cfg.clone(),
                load_proc_macro,
                load,
                project,
                sysroot,
            ),
            ProjectWorkspace::Cargo {
                cargo,
                sysroot,
                rustc,
                rustc_cfg,
                cfg_overrides,
                build_scripts,
                toolchain: _,
            } => cargo_to_crate_graph(
                rustc_cfg.clone(),
                cfg_overrides,
                load_proc_macro,
                load,
                cargo,
                build_scripts,
                sysroot.as_ref(),
                rustc,
            ),
            ProjectWorkspace::DetachedFiles { files, sysroot, rustc_cfg } => {
                detached_files_to_crate_graph(rustc_cfg.clone(), load, files, sysroot)
            }
        };
        if crate_graph.patch_cfg_if() {
            tracing::debug!("Patched std to depend on cfg-if")
        } else {
            tracing::debug!("Did not patch std to depend on cfg-if")
        }
        crate_graph
    }
}

fn project_json_to_crate_graph(
    rustc_cfg: Vec<CfgFlag>,
    load_proc_macro: &mut dyn FnMut(&str, &AbsPath) -> ProcMacroLoadResult,
    load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
    project: &ProjectJson,
    sysroot: &Option<Sysroot>,
) -> CrateGraph {
    let mut crate_graph = CrateGraph::default();
    let sysroot_deps = sysroot
        .as_ref()
        .map(|sysroot| sysroot_to_crate_graph(&mut crate_graph, sysroot, rustc_cfg.clone(), load));

    let mut cfg_cache: FxHashMap<&str, Vec<CfgFlag>> = FxHashMap::default();
    let crates: FxHashMap<CrateId, CrateId> = project
        .crates()
        .filter_map(|(crate_id, krate)| {
            let file_path = &krate.root_module;
            let file_id = load(file_path)?;
            Some((crate_id, krate, file_id))
        })
        .map(|(crate_id, krate, file_id)| {
            let env = krate.env.clone().into_iter().collect();
            let proc_macro = match krate.proc_macro_dylib_path.clone() {
                Some(it) => load_proc_macro(
                    krate.display_name.as_ref().map(|it| it.canonical_name()).unwrap_or(""),
                    &it,
                ),
                None => Err("no proc macro dylib present".into()),
            };

            let target_cfgs = match krate.target.as_deref() {
                Some(target) => {
                    cfg_cache.entry(target).or_insert_with(|| rustc_cfg::get(None, Some(target)))
                }
                None => &rustc_cfg,
            };

            let mut cfg_options = CfgOptions::default();
            cfg_options.extend(target_cfgs.iter().chain(krate.cfg.iter()).cloned());
            (
                crate_id,
                crate_graph.add_crate_root(
                    file_id,
                    krate.edition,
                    krate.display_name.clone(),
                    krate.version.clone(),
                    cfg_options.clone(),
                    cfg_options,
                    env,
                    proc_macro,
                    krate.is_proc_macro,
                    if krate.display_name.is_some() {
                        CrateOrigin::CratesIo { repo: krate.repository.clone() }
                    } else {
                        CrateOrigin::CratesIo { repo: None }
                    },
                ),
            )
        })
        .collect();

    for (from, krate) in project.crates() {
        if let Some(&from) = crates.get(&from) {
            if let Some((public_deps, libproc_macro)) = &sysroot_deps {
                public_deps.add(from, &mut crate_graph);
                if krate.is_proc_macro {
                    if let Some(proc_macro) = libproc_macro {
                        add_dep(
                            &mut crate_graph,
                            from,
                            CrateName::new("proc_macro").unwrap(),
                            *proc_macro,
                        );
                    }
                }
            }

            for dep in &krate.deps {
                if let Some(&to) = crates.get(&dep.crate_id) {
                    add_dep(&mut crate_graph, from, dep.name.clone(), to)
                }
            }
        }
    }
    crate_graph
}

fn cargo_to_crate_graph(
    rustc_cfg: Vec<CfgFlag>,
    override_cfg: &CfgOverrides,
    load_proc_macro: &mut dyn FnMut(&str, &AbsPath) -> ProcMacroLoadResult,
    load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
    cargo: &CargoWorkspace,
    build_scripts: &WorkspaceBuildScripts,
    sysroot: Option<&Sysroot>,
    rustc: &Option<CargoWorkspace>,
) -> CrateGraph {
    let _p = profile::span("cargo_to_crate_graph");
    let mut crate_graph = CrateGraph::default();
    let (public_deps, libproc_macro) = match sysroot {
        Some(sysroot) => sysroot_to_crate_graph(&mut crate_graph, sysroot, rustc_cfg.clone(), load),
        None => (SysrootPublicDeps::default(), None),
    };

    let mut cfg_options = CfgOptions::default();
    cfg_options.extend(rustc_cfg);

    let mut pkg_to_lib_crate = FxHashMap::default();

    cfg_options.insert_atom("debug_assertions".into());

    let mut pkg_crates = FxHashMap::default();
    // Does any crate signal to rust-analyzer that they need the rustc_private crates?
    let mut has_private = false;
    // Next, create crates for each package, target pair
    for pkg in cargo.packages() {
        let mut cfg_options = cfg_options.clone();

        let overrides = match override_cfg {
            CfgOverrides::Wildcard(cfg_diff) => Some(cfg_diff),
            CfgOverrides::Selective(cfg_overrides) => cfg_overrides.get(&cargo[pkg].name),
        };

        // Add test cfg for local crates
        if cargo[pkg].is_local {
            cfg_options.insert_atom("test".into());
        }

        if let Some(overrides) = overrides {
            // FIXME: this is sort of a hack to deal with #![cfg(not(test))] vanishing such as seen
            // in ed25519_dalek (#7243), and libcore (#9203) (although you only hit that one while
            // working on rust-lang/rust as that's the only time it appears outside sysroot).
            //
            // A more ideal solution might be to reanalyze crates based on where the cursor is and
            // figure out the set of cfgs that would have to apply to make it active.

            cfg_options.apply_diff(overrides.clone());
        };

        has_private |= cargo[pkg].metadata.rustc_private;
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

            if let Some(file_id) = load(&cargo[tgt].root) {
                let crate_id = add_target_crate_root(
                    &mut crate_graph,
                    &cargo[pkg],
                    build_scripts.get_output(pkg),
                    cfg_options.clone(),
                    &mut |path| load_proc_macro(&cargo[tgt].name, path),
                    file_id,
                    &cargo[tgt].name,
                    cargo[tgt].is_proc_macro,
                );
                if cargo[tgt].kind == TargetKind::Lib {
                    lib_tgt = Some((crate_id, cargo[tgt].name.clone()));
                    pkg_to_lib_crate.insert(pkg, crate_id);
                }
                if let Some(proc_macro) = libproc_macro {
                    add_dep_with_prelude(
                        &mut crate_graph,
                        crate_id,
                        CrateName::new("proc_macro").unwrap(),
                        proc_macro,
                        cargo[tgt].is_proc_macro,
                    );
                }

                pkg_crates.entry(pkg).or_insert_with(Vec::new).push((crate_id, cargo[tgt].kind));
            }
        }

        // Set deps to the core, std and to the lib target of the current package
        for (from, kind) in pkg_crates.get(&pkg).into_iter().flatten() {
            // Add sysroot deps first so that a lib target named `core` etc. can overwrite them.
            public_deps.add(*from, &mut crate_graph);

            if let Some((to, name)) = lib_tgt.clone() {
                if to != *from && *kind != TargetKind::BuildScript {
                    // (build script can not depend on its library target)

                    // For root projects with dashes in their name,
                    // cargo metadata does not do any normalization,
                    // so we do it ourselves currently
                    let name = CrateName::normalize_dashes(&name);
                    add_dep(&mut crate_graph, *from, name, to);
                }
            }
        }
    }

    // Now add a dep edge from all targets of upstream to the lib
    // target of downstream.
    for pkg in cargo.packages() {
        for dep in cargo[pkg].dependencies.iter() {
            let name = CrateName::new(&dep.name).unwrap();
            if let Some(&to) = pkg_to_lib_crate.get(&dep.pkg) {
                for (from, kind) in pkg_crates.get(&pkg).into_iter().flatten() {
                    if dep.kind == DepKind::Build && *kind != TargetKind::BuildScript {
                        // Only build scripts may depend on build dependencies.
                        continue;
                    }
                    if dep.kind != DepKind::Build && *kind == TargetKind::BuildScript {
                        // Build scripts may only depend on build dependencies.
                        continue;
                    }

                    add_dep(&mut crate_graph, *from, name.clone(), to)
                }
            }
        }
    }

    if has_private {
        // If the user provided a path to rustc sources, we add all the rustc_private crates
        // and create dependencies on them for the crates which opt-in to that
        if let Some(rustc_workspace) = rustc {
            handle_rustc_crates(
                rustc_workspace,
                load,
                &mut crate_graph,
                &cfg_options,
                override_cfg,
                load_proc_macro,
                &mut pkg_to_lib_crate,
                &public_deps,
                cargo,
                &pkg_crates,
                build_scripts,
            );
        }
    }
    crate_graph
}

fn detached_files_to_crate_graph(
    rustc_cfg: Vec<CfgFlag>,
    load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
    detached_files: &[AbsPathBuf],
    sysroot: &Sysroot,
) -> CrateGraph {
    let _p = profile::span("detached_files_to_crate_graph");
    let mut crate_graph = CrateGraph::default();
    let (public_deps, _libproc_macro) =
        sysroot_to_crate_graph(&mut crate_graph, sysroot, rustc_cfg.clone(), load);

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
            display_name,
            None,
            cfg_options.clone(),
            cfg_options.clone(),
            Env::default(),
            Ok(Vec::new()),
            false,
            CrateOrigin::CratesIo { repo: None },
        );

        public_deps.add(detached_file_crate, &mut crate_graph);
    }
    crate_graph
}

fn handle_rustc_crates(
    rustc_workspace: &CargoWorkspace,
    load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
    crate_graph: &mut CrateGraph,
    cfg_options: &CfgOptions,
    override_cfg: &CfgOverrides,
    load_proc_macro: &mut dyn FnMut(&str, &AbsPath) -> ProcMacroLoadResult,
    pkg_to_lib_crate: &mut FxHashMap<la_arena::Idx<crate::PackageData>, CrateId>,
    public_deps: &SysrootPublicDeps,
    cargo: &CargoWorkspace,
    pkg_crates: &FxHashMap<la_arena::Idx<crate::PackageData>, Vec<(CrateId, TargetKind)>>,
    build_scripts: &WorkspaceBuildScripts,
) {
    let mut rustc_pkg_crates = FxHashMap::default();
    // The root package of the rustc-dev component is rustc_driver, so we match that
    let root_pkg =
        rustc_workspace.packages().find(|package| rustc_workspace[*package].name == "rustc_driver");
    // The rustc workspace might be incomplete (such as if rustc-dev is not
    // installed for the current toolchain) and `rustc_source` is set to discover.
    if let Some(root_pkg) = root_pkg {
        // Iterate through every crate in the dependency subtree of rustc_driver using BFS
        let mut queue = VecDeque::new();
        queue.push_back(root_pkg);
        while let Some(pkg) = queue.pop_front() {
            // Don't duplicate packages if they are dependended on a diamond pattern
            // N.B. if this line is omitted, we try to analyse over 4_800_000 crates
            // which is not ideal
            if rustc_pkg_crates.contains_key(&pkg) {
                continue;
            }
            for dep in &rustc_workspace[pkg].dependencies {
                queue.push_back(dep.pkg);
            }

            let mut cfg_options = cfg_options.clone();

            let overrides = match override_cfg {
                CfgOverrides::Wildcard(cfg_diff) => Some(cfg_diff),
                CfgOverrides::Selective(cfg_overrides) => {
                    cfg_overrides.get(&rustc_workspace[pkg].name)
                }
            };

            if let Some(overrides) = overrides {
                // FIXME: this is sort of a hack to deal with #![cfg(not(test))] vanishing such as seen
                // in ed25519_dalek (#7243), and libcore (#9203) (although you only hit that one while
                // working on rust-lang/rust as that's the only time it appears outside sysroot).
                //
                // A more ideal solution might be to reanalyze crates based on where the cursor is and
                // figure out the set of cfgs that would have to apply to make it active.

                cfg_options.apply_diff(overrides.clone());
            };

            for &tgt in rustc_workspace[pkg].targets.iter() {
                if rustc_workspace[tgt].kind != TargetKind::Lib {
                    continue;
                }
                if let Some(file_id) = load(&rustc_workspace[tgt].root) {
                    let crate_id = add_target_crate_root(
                        crate_graph,
                        &rustc_workspace[pkg],
                        build_scripts.get_output(pkg),
                        cfg_options.clone(),
                        &mut |path| load_proc_macro(&rustc_workspace[tgt].name, path),
                        file_id,
                        &rustc_workspace[tgt].name,
                        rustc_workspace[tgt].is_proc_macro,
                    );
                    pkg_to_lib_crate.insert(pkg, crate_id);
                    // Add dependencies on core / std / alloc for this crate
                    public_deps.add(crate_id, crate_graph);
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
    pkg: &PackageData,
    build_data: Option<&BuildScriptOutput>,
    cfg_options: CfgOptions,
    load_proc_macro: &mut dyn FnMut(&AbsPath) -> ProcMacroLoadResult,
    file_id: FileId,
    cargo_name: &str,
    is_proc_macro: bool,
) -> CrateId {
    let edition = pkg.edition;
    let mut potential_cfg_options = cfg_options.clone();
    potential_cfg_options.extend(
        pkg.features
            .iter()
            .map(|feat| CfgFlag::KeyValue { key: "feature".into(), value: feat.0.into() }),
    );
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

    let proc_macro = match build_data.as_ref().map(|it| it.proc_macro_dylib_path.as_ref()) {
        Some(Some(it)) => load_proc_macro(it),
        Some(None) => Err("no proc macro dylib present".into()),
        None => Err("crate has not (yet) been built".into()),
    };

    let display_name = CrateDisplayName::from_canonical_name(cargo_name.to_string());
    crate_graph.add_crate_root(
        file_id,
        edition,
        Some(display_name),
        Some(pkg.version.to_string()),
        cfg_options,
        potential_cfg_options,
        env,
        proc_macro,
        is_proc_macro,
        CrateOrigin::CratesIo { repo: pkg.repository.clone() },
    )
}

#[derive(Default)]
struct SysrootPublicDeps {
    deps: Vec<(CrateName, CrateId, bool)>,
}

impl SysrootPublicDeps {
    /// Makes `from` depend on the public sysroot crates.
    fn add(&self, from: CrateId, crate_graph: &mut CrateGraph) {
        for (name, krate, prelude) in &self.deps {
            add_dep_with_prelude(crate_graph, from, name.clone(), *krate, *prelude);
        }
    }
}

fn sysroot_to_crate_graph(
    crate_graph: &mut CrateGraph,
    sysroot: &Sysroot,
    rustc_cfg: Vec<CfgFlag>,
    load: &mut dyn FnMut(&AbsPath) -> Option<FileId>,
) -> (SysrootPublicDeps, Option<CrateId>) {
    let _p = profile::span("sysroot_to_crate_graph");
    let mut cfg_options = CfgOptions::default();
    cfg_options.extend(rustc_cfg);
    let sysroot_crates: FxHashMap<SysrootCrate, CrateId> = sysroot
        .crates()
        .filter_map(|krate| {
            let file_id = load(&sysroot[krate].root)?;

            let env = Env::default();
            let display_name = CrateDisplayName::from_canonical_name(sysroot[krate].name.clone());
            let crate_id = crate_graph.add_crate_root(
                file_id,
                Edition::CURRENT,
                Some(display_name),
                None,
                cfg_options.clone(),
                cfg_options.clone(),
                env,
                Err("no proc macro loaded for sysroot crate".into()),
                false,
                CrateOrigin::Lang(LangCrateOrigin::from(&*sysroot[krate].name)),
            );
            Some((krate, crate_id))
        })
        .collect();

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
            .map(|(name, idx, prelude)| {
                (CrateName::new(name).unwrap(), sysroot_crates[&idx], prelude)
            })
            .collect::<Vec<_>>(),
    };

    let libproc_macro = sysroot.proc_macro().and_then(|it| sysroot_crates.get(&it).copied());
    (public_deps, libproc_macro)
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
