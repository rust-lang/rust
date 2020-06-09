//! FIXME: write short doc here

mod cargo_workspace;
mod json_project;
mod sysroot;

use std::{
    fs::{read_dir, File, ReadDir},
    io::{self, BufReader},
    path::{Path, PathBuf},
    process::{Command, Output},
};

use anyhow::{bail, Context, Result};
use ra_cfg::CfgOptions;
use ra_db::{CrateGraph, CrateName, Edition, Env, ExternSource, ExternSourceId, FileId};
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::from_reader;

pub use crate::{
    cargo_workspace::{CargoConfig, CargoWorkspace, Package, Target, TargetKind},
    json_project::JsonProject,
    sysroot::Sysroot,
};
pub use ra_proc_macro::ProcMacroClient;

#[derive(Debug, Clone)]
pub enum ProjectWorkspace {
    /// Project workspace was discovered by running `cargo metadata` and `rustc --print sysroot`.
    Cargo { cargo: CargoWorkspace, sysroot: Sysroot },
    /// Project workspace was manually specified using a `rust-project.json` file.
    Json { project: JsonProject, project_location: PathBuf },
}

/// `PackageRoot` describes a package root folder.
/// Which may be an external dependency, or a member of
/// the current workspace.
#[derive(Debug, Clone)]
pub struct PackageRoot {
    /// Path to the root folder
    path: PathBuf,
    /// Is a member of the current workspace
    is_member: bool,
    out_dir: Option<PathBuf>,
}
impl PackageRoot {
    pub fn new_member(path: PathBuf) -> PackageRoot {
        Self { path, is_member: true, out_dir: None }
    }
    pub fn new_non_member(path: PathBuf) -> PackageRoot {
        Self { path, is_member: false, out_dir: None }
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn out_dir(&self) -> Option<&Path> {
        self.out_dir.as_deref()
    }
    pub fn is_member(&self) -> bool {
        self.is_member
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum ProjectManifest {
    ProjectJson(PathBuf),
    CargoToml(PathBuf),
}

impl ProjectManifest {
    pub fn from_manifest_file(path: PathBuf) -> Result<ProjectManifest> {
        if path.ends_with("rust-project.json") {
            return Ok(ProjectManifest::ProjectJson(path));
        }
        if path.ends_with("Cargo.toml") {
            return Ok(ProjectManifest::CargoToml(path));
        }
        bail!("project root must point to Cargo.toml or rust-project.json: {}", path.display())
    }

    pub fn discover_single(path: &Path) -> Result<ProjectManifest> {
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

    pub fn discover(path: &Path) -> io::Result<Vec<ProjectManifest>> {
        if let Some(project_json) = find_in_parent_dirs(path, "rust-project.json") {
            return Ok(vec![ProjectManifest::ProjectJson(project_json)]);
        }
        return find_cargo_toml(path)
            .map(|paths| paths.into_iter().map(ProjectManifest::CargoToml).collect());

        fn find_cargo_toml(path: &Path) -> io::Result<Vec<PathBuf>> {
            match find_in_parent_dirs(path, "Cargo.toml") {
                Some(it) => Ok(vec![it]),
                None => Ok(find_cargo_toml_in_child_dir(read_dir(path)?)),
            }
        }

        fn find_in_parent_dirs(path: &Path, target_file_name: &str) -> Option<PathBuf> {
            if path.ends_with(target_file_name) {
                return Some(path.to_owned());
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

        fn find_cargo_toml_in_child_dir(entities: ReadDir) -> Vec<PathBuf> {
            // Only one level down to avoid cycles the easy way and stop a runaway scan with large projects
            entities
                .filter_map(Result::ok)
                .map(|it| it.path().join("Cargo.toml"))
                .filter(|it| it.exists())
                .collect()
        }
    }

    pub fn discover_all(paths: &[impl AsRef<Path>]) -> Vec<ProjectManifest> {
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

impl ProjectWorkspace {
    pub fn load(
        manifest: ProjectManifest,
        cargo_features: &CargoConfig,
        with_sysroot: bool,
    ) -> Result<ProjectWorkspace> {
        let res = match manifest {
            ProjectManifest::ProjectJson(project_json) => {
                let file = File::open(&project_json).with_context(|| {
                    format!("Failed to open json file {}", project_json.display())
                })?;
                let reader = BufReader::new(file);
                let project_location = match project_json.parent() {
                    Some(parent) => PathBuf::from(parent),
                    None => PathBuf::new(),
                };
                ProjectWorkspace::Json {
                    project: from_reader(reader).with_context(|| {
                        format!("Failed to deserialize json file {}", project_json.display())
                    })?,
                    project_location: project_location,
                }
            }
            ProjectManifest::CargoToml(cargo_toml) => {
                let cargo = CargoWorkspace::from_cargo_metadata(&cargo_toml, cargo_features)
                    .with_context(|| {
                        format!(
                            "Failed to read Cargo metadata from Cargo.toml file {}",
                            cargo_toml.display()
                        )
                    })?;
                let sysroot = if with_sysroot {
                    Sysroot::discover(&cargo_toml).with_context(|| {
                        format!(
                            "Failed to find sysroot for Cargo.toml file {}. Is rust-src installed?",
                            cargo_toml.display()
                        )
                    })?
                } else {
                    Sysroot::default()
                };
                ProjectWorkspace::Cargo { cargo, sysroot }
            }
        };

        Ok(res)
    }

    /// Returns the roots for the current `ProjectWorkspace`
    /// The return type contains the path and whether or not
    /// the root is a member of the current workspace
    pub fn to_roots(&self) -> Vec<PackageRoot> {
        match self {
            ProjectWorkspace::Json { project, project_location } => project
                .roots
                .iter()
                .map(|r| PackageRoot::new_member(project_location.join(&r.path)))
                .collect(),
            ProjectWorkspace::Cargo { cargo, sysroot } => cargo
                .packages()
                .map(|pkg| PackageRoot {
                    path: cargo[pkg].root().to_path_buf(),
                    is_member: cargo[pkg].is_member,
                    out_dir: cargo[pkg].out_dir.clone(),
                })
                .chain(sysroot.crates().map(|krate| {
                    PackageRoot::new_non_member(sysroot[krate].root_dir().to_path_buf())
                }))
                .collect(),
        }
    }

    pub fn proc_macro_dylib_paths(&self) -> Vec<PathBuf> {
        match self {
            ProjectWorkspace::Json { project, .. } => project
                .crates
                .iter()
                .filter_map(|krate| krate.proc_macro_dylib_path.as_ref())
                .cloned()
                .collect(),
            ProjectWorkspace::Cargo { cargo, sysroot: _sysroot } => cargo
                .packages()
                .filter_map(|pkg| cargo[pkg].proc_macro_dylib_path.as_ref())
                .cloned()
                .collect(),
        }
    }

    pub fn n_packages(&self) -> usize {
        match self {
            ProjectWorkspace::Json { project, .. } => project.crates.len(),
            ProjectWorkspace::Cargo { cargo, sysroot } => {
                cargo.packages().len() + sysroot.crates().len()
            }
        }
    }

    pub fn to_crate_graph(
        &self,
        target: Option<&str>,
        extern_source_roots: &FxHashMap<PathBuf, ExternSourceId>,
        proc_macro_client: &ProcMacroClient,
        load: &mut dyn FnMut(&Path) -> Option<FileId>,
    ) -> CrateGraph {
        let mut crate_graph = CrateGraph::default();
        match self {
            ProjectWorkspace::Json { project, project_location } => {
                let crates: FxHashMap<_, _> = project
                    .crates
                    .iter()
                    .enumerate()
                    .filter_map(|(seq_index, krate)| {
                        let file_path = project_location.join(&krate.root_module);
                        let file_id = load(&file_path)?;
                        let edition = match krate.edition {
                            json_project::Edition::Edition2015 => Edition::Edition2015,
                            json_project::Edition::Edition2018 => Edition::Edition2018,
                        };
                        let cfg_options = {
                            let mut opts = CfgOptions::default();
                            for cfg in &krate.cfg {
                                match cfg.find('=') {
                                    None => opts.insert_atom(cfg.into()),
                                    Some(pos) => {
                                        let key = &cfg[..pos];
                                        let value = cfg[pos + 1..].trim_matches('"');
                                        opts.insert_key_value(key.into(), value.into());
                                    }
                                }
                            }
                            opts
                        };

                        let mut env = Env::default();
                        let mut extern_source = ExternSource::default();
                        if let Some(out_dir) = &krate.out_dir {
                            // NOTE: cargo and rustc seem to hide non-UTF-8 strings from env! and option_env!()
                            if let Some(out_dir) = out_dir.to_str().map(|s| s.to_owned()) {
                                env.set("OUT_DIR", out_dir);
                            }
                            if let Some(&extern_source_id) = extern_source_roots.get(out_dir) {
                                extern_source.set_extern_path(&out_dir, extern_source_id);
                            }
                        }
                        let proc_macro = krate
                            .proc_macro_dylib_path
                            .clone()
                            .map(|it| proc_macro_client.by_dylib_path(&it));
                        // FIXME: No crate name in json definition such that we cannot add OUT_DIR to env
                        Some((
                            json_project::CrateId(seq_index),
                            crate_graph.add_crate_root(
                                file_id,
                                edition,
                                // FIXME json definitions can store the crate name
                                None,
                                cfg_options,
                                env,
                                extern_source,
                                proc_macro.unwrap_or_default(),
                            ),
                        ))
                    })
                    .collect();

                for (id, krate) in project.crates.iter().enumerate() {
                    for dep in &krate.deps {
                        let from_crate_id = json_project::CrateId(id);
                        let to_crate_id = dep.krate;
                        if let (Some(&from), Some(&to)) =
                            (crates.get(&from_crate_id), crates.get(&to_crate_id))
                        {
                            if crate_graph
                                .add_dep(from, CrateName::new(&dep.name).unwrap(), to)
                                .is_err()
                            {
                                log::error!(
                                    "cyclic dependency {:?} -> {:?}",
                                    from_crate_id,
                                    to_crate_id
                                );
                            }
                        }
                    }
                }
            }
            ProjectWorkspace::Cargo { cargo, sysroot } => {
                let mut cfg_options = get_rustc_cfg_options(target);

                let sysroot_crates: FxHashMap<_, _> = sysroot
                    .crates()
                    .filter_map(|krate| {
                        let file_id = load(&sysroot[krate].root)?;

                        let env = Env::default();
                        let extern_source = ExternSource::default();
                        let proc_macro = vec![];
                        let crate_name = CrateName::new(&sysroot[krate].name)
                            .expect("Sysroot crate names should not contain dashes");

                        let crate_id = crate_graph.add_crate_root(
                            file_id,
                            Edition::Edition2018,
                            Some(crate_name),
                            cfg_options.clone(),
                            env,
                            extern_source,
                            proc_macro,
                        );
                        Some((krate, crate_id))
                    })
                    .collect();

                for from in sysroot.crates() {
                    for &to in sysroot[from].deps.iter() {
                        let name = &sysroot[to].name;
                        if let (Some(&from), Some(&to)) =
                            (sysroot_crates.get(&from), sysroot_crates.get(&to))
                        {
                            if crate_graph.add_dep(from, CrateName::new(name).unwrap(), to).is_err()
                            {
                                log::error!("cyclic dependency between sysroot crates")
                            }
                        }
                    }
                }

                let libcore = sysroot.core().and_then(|it| sysroot_crates.get(&it).copied());
                let liballoc = sysroot.alloc().and_then(|it| sysroot_crates.get(&it).copied());
                let libstd = sysroot.std().and_then(|it| sysroot_crates.get(&it).copied());
                let libproc_macro =
                    sysroot.proc_macro().and_then(|it| sysroot_crates.get(&it).copied());

                let mut pkg_to_lib_crate = FxHashMap::default();
                let mut pkg_crates = FxHashMap::default();

                // Add test cfg for non-sysroot crates
                cfg_options.insert_atom("test".into());

                // Next, create crates for each package, target pair
                for pkg in cargo.packages() {
                    let mut lib_tgt = None;
                    for &tgt in cargo[pkg].targets.iter() {
                        let root = cargo[tgt].root.as_path();
                        if let Some(file_id) = load(root) {
                            let edition = cargo[pkg].edition;
                            let cfg_options = {
                                let mut opts = cfg_options.clone();
                                for feature in cargo[pkg].features.iter() {
                                    opts.insert_key_value("feature".into(), feature.into());
                                }
                                for cfg in cargo[pkg].cfgs.iter() {
                                    match cfg.find('=') {
                                        Some(split) => opts.insert_key_value(
                                            cfg[..split].into(),
                                            cfg[split + 1..].trim_matches('"').into(),
                                        ),
                                        None => opts.insert_atom(cfg.into()),
                                    };
                                }
                                opts
                            };
                            let mut env = Env::default();
                            let mut extern_source = ExternSource::default();
                            if let Some(out_dir) = &cargo[pkg].out_dir {
                                // NOTE: cargo and rustc seem to hide non-UTF-8 strings from env! and option_env!()
                                if let Some(out_dir) = out_dir.to_str().map(|s| s.to_owned()) {
                                    env.set("OUT_DIR", out_dir);
                                }
                                if let Some(&extern_source_id) = extern_source_roots.get(out_dir) {
                                    extern_source.set_extern_path(&out_dir, extern_source_id);
                                }
                            }
                            let proc_macro = cargo[pkg]
                                .proc_macro_dylib_path
                                .as_ref()
                                .map(|it| proc_macro_client.by_dylib_path(&it))
                                .unwrap_or_default();

                            let crate_id = crate_graph.add_crate_root(
                                file_id,
                                edition,
                                Some(CrateName::normalize_dashes(&cargo[pkg].name)),
                                cfg_options,
                                env,
                                extern_source,
                                proc_macro.clone(),
                            );
                            if cargo[tgt].kind == TargetKind::Lib {
                                lib_tgt = Some((crate_id, cargo[tgt].name.clone()));
                                pkg_to_lib_crate.insert(pkg, crate_id);
                            }
                            if cargo[tgt].is_proc_macro {
                                if let Some(proc_macro) = libproc_macro {
                                    if crate_graph
                                        .add_dep(
                                            crate_id,
                                            CrateName::new("proc_macro").unwrap(),
                                            proc_macro,
                                        )
                                        .is_err()
                                    {
                                        log::error!(
                                            "cyclic dependency on proc_macro for {}",
                                            &cargo[pkg].name
                                        )
                                    }
                                }
                            }

                            pkg_crates.entry(pkg).or_insert_with(Vec::new).push(crate_id);
                        }
                    }

                    // Set deps to the core, std and to the lib target of the current package
                    for &from in pkg_crates.get(&pkg).into_iter().flatten() {
                        if let Some((to, name)) = lib_tgt.clone() {
                            if to != from
                                && crate_graph
                                    .add_dep(
                                        from,
                                        // For root projects with dashes in their name,
                                        // cargo metadata does not do any normalization,
                                        // so we do it ourselves currently
                                        CrateName::normalize_dashes(&name),
                                        to,
                                    )
                                    .is_err()
                            {
                                {
                                    log::error!(
                                        "cyclic dependency between targets of {}",
                                        &cargo[pkg].name
                                    )
                                }
                            }
                        }
                        // core is added as a dependency before std in order to
                        // mimic rustcs dependency order
                        if let Some(core) = libcore {
                            if crate_graph
                                .add_dep(from, CrateName::new("core").unwrap(), core)
                                .is_err()
                            {
                                log::error!("cyclic dependency on core for {}", &cargo[pkg].name)
                            }
                        }
                        if let Some(alloc) = liballoc {
                            if crate_graph
                                .add_dep(from, CrateName::new("alloc").unwrap(), alloc)
                                .is_err()
                            {
                                log::error!("cyclic dependency on alloc for {}", &cargo[pkg].name)
                            }
                        }
                        if let Some(std) = libstd {
                            if crate_graph
                                .add_dep(from, CrateName::new("std").unwrap(), std)
                                .is_err()
                            {
                                log::error!("cyclic dependency on std for {}", &cargo[pkg].name)
                            }
                        }
                    }
                }

                // Now add a dep edge from all targets of upstream to the lib
                // target of downstream.
                for pkg in cargo.packages() {
                    for dep in cargo[pkg].dependencies.iter() {
                        if let Some(&to) = pkg_to_lib_crate.get(&dep.pkg) {
                            for &from in pkg_crates.get(&pkg).into_iter().flatten() {
                                if crate_graph
                                    .add_dep(from, CrateName::new(&dep.name).unwrap(), to)
                                    .is_err()
                                {
                                    log::error!(
                                        "cyclic dependency {} -> {}",
                                        &cargo[pkg].name,
                                        &cargo[dep.pkg].name
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
        crate_graph
    }

    pub fn workspace_root_for(&self, path: &Path) -> Option<&Path> {
        match self {
            ProjectWorkspace::Cargo { cargo, .. } => {
                Some(cargo.workspace_root()).filter(|root| path.starts_with(root))
            }
            ProjectWorkspace::Json { project: JsonProject { roots, .. }, .. } => roots
                .iter()
                .find(|root| path.starts_with(&root.path))
                .map(|root| root.path.as_ref()),
        }
    }
}

fn get_rustc_cfg_options(target: Option<&str>) -> CfgOptions {
    let mut cfg_options = CfgOptions::default();

    // Some nightly-only cfgs, which are required for stdlib
    {
        cfg_options.insert_atom("target_thread_local".into());
        for &target_has_atomic in ["8", "16", "32", "64", "cas", "ptr"].iter() {
            cfg_options.insert_key_value("target_has_atomic".into(), target_has_atomic.into());
            cfg_options
                .insert_key_value("target_has_atomic_load_store".into(), target_has_atomic.into());
        }
    }

    let rustc_cfgs = || -> Result<String> {
        // `cfg(test)` and `cfg(debug_assertion)` are handled outside, so we suppress them here.
        let mut cmd = Command::new(ra_toolchain::rustc());
        cmd.args(&["--print", "cfg", "-O"]);
        if let Some(target) = target {
            cmd.args(&["--target", target]);
        }
        let output = output(cmd)?;
        Ok(String::from_utf8(output.stdout)?)
    }();

    match rustc_cfgs {
        Ok(rustc_cfgs) => {
            for line in rustc_cfgs.lines() {
                match line.find('=') {
                    None => cfg_options.insert_atom(line.into()),
                    Some(pos) => {
                        let key = &line[..pos];
                        let value = line[pos + 1..].trim_matches('"');
                        cfg_options.insert_key_value(key.into(), value.into());
                    }
                }
            }
        }
        Err(e) => log::error!("failed to get rustc cfgs: {:#}", e),
    }

    cfg_options.insert_atom("debug_assertions".into());

    cfg_options
}

fn output(mut cmd: Command) -> Result<Output> {
    let output = cmd.output().with_context(|| format!("{:?} failed", cmd))?;
    if !output.status.success() {
        match String::from_utf8(output.stderr) {
            Ok(stderr) if !stderr.is_empty() => {
                bail!("{:?} failed, {}\nstderr:\n{}", cmd, output.status, stderr)
            }
            _ => bail!("{:?} failed, {}", cmd, output.status),
        }
    }
    Ok(output)
}
