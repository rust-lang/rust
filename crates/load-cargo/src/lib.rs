//! Loads a Cargo project into a static instance of analysis, without support
//! for incorporating changes.
// Note, don't remove any public api from this. This API is consumed by external tools
// to run rust-analyzer as a library.
use std::{collections::hash_map::Entry, mem, path::Path, sync};

use ::tt::token_id as tt;
use crossbeam_channel::{unbounded, Receiver};
use ide::{AnalysisHost, Change, SourceRoot};
use ide_db::{
    base_db::{
        CrateGraph, Env, ProcMacro, ProcMacroExpander, ProcMacroExpansionError, ProcMacroKind,
        ProcMacroLoadResult, ProcMacros,
    },
    FxHashMap,
};
use itertools::Itertools;
use proc_macro_api::{MacroDylib, ProcMacroServer};
use project_model::{CargoConfig, PackageRoot, ProjectManifest, ProjectWorkspace};
use vfs::{file_set::FileSetConfig, loader::Handle, AbsPath, AbsPathBuf, VfsPath};

pub struct LoadCargoConfig {
    pub load_out_dirs_from_check: bool,
    pub with_proc_macro_server: ProcMacroServerChoice,
    pub prefill_caches: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcMacroServerChoice {
    Sysroot,
    Explicit(AbsPathBuf),
    None,
}

pub fn load_workspace_at(
    root: &Path,
    cargo_config: &CargoConfig,
    load_config: &LoadCargoConfig,
    progress: &dyn Fn(String),
) -> anyhow::Result<(AnalysisHost, vfs::Vfs, Option<ProcMacroServer>)> {
    let root = AbsPathBuf::assert(std::env::current_dir()?.join(root));
    let root = ProjectManifest::discover_single(&root)?;
    let mut workspace = ProjectWorkspace::load(root, cargo_config, progress)?;

    if load_config.load_out_dirs_from_check {
        let build_scripts = workspace.run_build_scripts(cargo_config, progress)?;
        workspace.set_build_scripts(build_scripts)
    }

    load_workspace(workspace, &cargo_config.extra_env, load_config)
}

pub fn load_workspace(
    ws: ProjectWorkspace,
    extra_env: &FxHashMap<String, String>,
    load_config: &LoadCargoConfig,
) -> anyhow::Result<(AnalysisHost, vfs::Vfs, Option<ProcMacroServer>)> {
    let (sender, receiver) = unbounded();
    let mut vfs = vfs::Vfs::default();
    let mut loader = {
        let loader =
            vfs_notify::NotifyHandle::spawn(Box::new(move |msg| sender.send(msg).unwrap()));
        Box::new(loader)
    };

    let proc_macro_server = match &load_config.with_proc_macro_server {
        ProcMacroServerChoice::Sysroot => ws
            .find_sysroot_proc_macro_srv()
            .and_then(|it| ProcMacroServer::spawn(it).map_err(Into::into)),
        ProcMacroServerChoice::Explicit(path) => {
            ProcMacroServer::spawn(path.clone()).map_err(Into::into)
        }
        ProcMacroServerChoice::None => Err(anyhow::format_err!("proc macro server disabled")),
    };

    let (crate_graph, proc_macros) = ws.to_crate_graph(
        &mut |path: &AbsPath| {
            let contents = loader.load_sync(path);
            let path = vfs::VfsPath::from(path.to_path_buf());
            vfs.set_file_contents(path.clone(), contents);
            vfs.file_id(&path)
        },
        extra_env,
    );
    let proc_macros = {
        let proc_macro_server = match &proc_macro_server {
            Ok(it) => Ok(it),
            Err(e) => Err(e.to_string()),
        };
        proc_macros
            .into_iter()
            .map(|(crate_id, path)| {
                (
                    crate_id,
                    path.map_or_else(
                        |_| Err("proc macro crate is missing dylib".to_owned()),
                        |(_, path)| {
                            proc_macro_server.as_ref().map_err(Clone::clone).and_then(
                                |proc_macro_server| load_proc_macro(proc_macro_server, &path, &[]),
                            )
                        },
                    ),
                )
            })
            .collect()
    };

    let project_folders = ProjectFolders::new(&[ws], &[]);
    loader.set_config(vfs::loader::Config {
        load: project_folders.load,
        watch: vec![],
        version: 0,
    });

    let host = load_crate_graph(
        crate_graph,
        proc_macros,
        project_folders.source_root_config,
        &mut vfs,
        &receiver,
    );

    if load_config.prefill_caches {
        host.analysis().parallel_prime_caches(1, |_| {})?;
    }
    Ok((host, vfs, proc_macro_server.ok()))
}

#[derive(Default)]
pub struct ProjectFolders {
    pub load: Vec<vfs::loader::Entry>,
    pub watch: Vec<usize>,
    pub source_root_config: SourceRootConfig,
}

impl ProjectFolders {
    pub fn new(workspaces: &[ProjectWorkspace], global_excludes: &[AbsPathBuf]) -> ProjectFolders {
        let mut res = ProjectFolders::default();
        let mut fsc = FileSetConfig::builder();
        let mut local_filesets = vec![];

        // Dedup source roots
        // Depending on the project setup, we can have duplicated source roots, or for example in
        // the case of the rustc workspace, we can end up with two source roots that are almost the
        // same but not quite, like:
        // PackageRoot { is_local: false, include: [AbsPathBuf(".../rust/src/tools/miri/cargo-miri")], exclude: [] }
        // PackageRoot {
        //     is_local: true,
        //     include: [AbsPathBuf(".../rust/src/tools/miri/cargo-miri"), AbsPathBuf(".../rust/build/x86_64-pc-windows-msvc/stage0-tools/x86_64-pc-windows-msvc/release/build/cargo-miri-85801cd3d2d1dae4/out")],
        //     exclude: [AbsPathBuf(".../rust/src/tools/miri/cargo-miri/.git"), AbsPathBuf(".../rust/src/tools/miri/cargo-miri/target")]
        // }
        //
        // The first one comes from the explicit rustc workspace which points to the rustc workspace itself
        // The second comes from the rustc workspace that we load as the actual project workspace
        // These `is_local` differing in this kind of way gives us problems, especially when trying to filter diagnostics as we don't report diagnostics for external libraries.
        // So we need to deduplicate these, usually it would be enough to deduplicate by `include`, but as the rustc example shows here that doesn't work,
        // so we need to also coalesce the includes if they overlap.

        let mut roots: Vec<_> = workspaces
            .iter()
            .flat_map(|ws| ws.to_roots())
            .update(|root| root.include.sort())
            .sorted_by(|a, b| a.include.cmp(&b.include))
            .collect();

        // map that tracks indices of overlapping roots
        let mut overlap_map = FxHashMap::<_, Vec<_>>::default();
        let mut done = false;

        while !mem::replace(&mut done, true) {
            // maps include paths to indices of the corresponding root
            let mut include_to_idx = FxHashMap::default();
            // Find and note down the indices of overlapping roots
            for (idx, root) in roots.iter().enumerate().filter(|(_, it)| !it.include.is_empty()) {
                for include in &root.include {
                    match include_to_idx.entry(include) {
                        Entry::Occupied(e) => {
                            overlap_map.entry(*e.get()).or_default().push(idx);
                        }
                        Entry::Vacant(e) => {
                            e.insert(idx);
                        }
                    }
                }
            }
            for (k, v) in overlap_map.drain() {
                done = false;
                for v in v {
                    let r = mem::replace(
                        &mut roots[v],
                        PackageRoot { is_local: false, include: vec![], exclude: vec![] },
                    );
                    roots[k].is_local |= r.is_local;
                    roots[k].include.extend(r.include);
                    roots[k].exclude.extend(r.exclude);
                }
                roots[k].include.sort();
                roots[k].exclude.sort();
                roots[k].include.dedup();
                roots[k].exclude.dedup();
            }
        }

        for root in roots.into_iter().filter(|it| !it.include.is_empty()) {
            let file_set_roots: Vec<VfsPath> =
                root.include.iter().cloned().map(VfsPath::from).collect();

            let entry = {
                let mut dirs = vfs::loader::Directories::default();
                dirs.extensions.push("rs".into());
                dirs.include.extend(root.include);
                dirs.exclude.extend(root.exclude);
                for excl in global_excludes {
                    if dirs
                        .include
                        .iter()
                        .any(|incl| incl.starts_with(excl) || excl.starts_with(incl))
                    {
                        dirs.exclude.push(excl.clone());
                    }
                }

                vfs::loader::Entry::Directories(dirs)
            };

            if root.is_local {
                res.watch.push(res.load.len());
            }
            res.load.push(entry);

            if root.is_local {
                local_filesets.push(fsc.len());
            }
            fsc.add_file_set(file_set_roots)
        }

        let fsc = fsc.build();
        res.source_root_config = SourceRootConfig { fsc, local_filesets };

        res
    }
}

#[derive(Default, Debug)]
pub struct SourceRootConfig {
    pub fsc: FileSetConfig,
    pub local_filesets: Vec<usize>,
}

impl SourceRootConfig {
    pub fn partition(&self, vfs: &vfs::Vfs) -> Vec<SourceRoot> {
        self.fsc
            .partition(vfs)
            .into_iter()
            .enumerate()
            .map(|(idx, file_set)| {
                let is_local = self.local_filesets.contains(&idx);
                if is_local {
                    SourceRoot::new_local(file_set)
                } else {
                    SourceRoot::new_library(file_set)
                }
            })
            .collect()
    }
}

/// Load the proc-macros for the given lib path, replacing all expanders whose names are in `dummy_replace`
/// with an identity dummy expander.
pub fn load_proc_macro(
    server: &ProcMacroServer,
    path: &AbsPath,
    dummy_replace: &[Box<str>],
) -> ProcMacroLoadResult {
    let res: Result<Vec<_>, String> = (|| {
        let dylib = MacroDylib::new(path.to_path_buf());
        let vec = server.load_dylib(dylib).map_err(|e| format!("{e}"))?;
        if vec.is_empty() {
            return Err("proc macro library returned no proc macros".to_string());
        }
        Ok(vec
            .into_iter()
            .map(|expander| expander_to_proc_macro(expander, dummy_replace))
            .collect())
    })();
    match res {
        Ok(proc_macros) => {
            tracing::info!(
                "Loaded proc-macros for {path}: {:?}",
                proc_macros.iter().map(|it| it.name.clone()).collect::<Vec<_>>()
            );
            Ok(proc_macros)
        }
        Err(e) => {
            tracing::warn!("proc-macro loading for {path} failed: {e}");
            Err(e)
        }
    }
}

fn load_crate_graph(
    crate_graph: CrateGraph,
    proc_macros: ProcMacros,
    source_root_config: SourceRootConfig,
    vfs: &mut vfs::Vfs,
    receiver: &Receiver<vfs::loader::Message>,
) -> AnalysisHost {
    let lru_cap = std::env::var("RA_LRU_CAP").ok().and_then(|it| it.parse::<usize>().ok());
    let mut host = AnalysisHost::new(lru_cap);
    let mut analysis_change = Change::new();

    host.raw_database_mut().enable_proc_attr_macros();

    // wait until Vfs has loaded all roots
    for task in receiver {
        match task {
            vfs::loader::Message::Progress { n_done, n_total, config_version: _ } => {
                if n_done == n_total {
                    break;
                }
            }
            vfs::loader::Message::Loaded { files } => {
                for (path, contents) in files {
                    vfs.set_file_contents(path.into(), contents);
                }
            }
        }
    }
    let changes = vfs.take_changes();
    for file in changes {
        if file.exists() {
            let contents = vfs.file_contents(file.file_id);
            if let Ok(text) = std::str::from_utf8(contents) {
                analysis_change.change_file(file.file_id, Some(text.into()))
            }
        }
    }
    let source_roots = source_root_config.partition(vfs);
    analysis_change.set_roots(source_roots);

    analysis_change.set_crate_graph(crate_graph);
    analysis_change.set_proc_macros(proc_macros);

    host.apply_change(analysis_change);
    host
}

fn expander_to_proc_macro(
    expander: proc_macro_api::ProcMacro,
    dummy_replace: &[Box<str>],
) -> ProcMacro {
    let name = From::from(expander.name());
    let kind = match expander.kind() {
        proc_macro_api::ProcMacroKind::CustomDerive => ProcMacroKind::CustomDerive,
        proc_macro_api::ProcMacroKind::FuncLike => ProcMacroKind::FuncLike,
        proc_macro_api::ProcMacroKind::Attr => ProcMacroKind::Attr,
    };
    let expander: sync::Arc<dyn ProcMacroExpander> =
        if dummy_replace.iter().any(|replace| &**replace == name) {
            match kind {
                ProcMacroKind::Attr => sync::Arc::new(IdentityExpander),
                _ => sync::Arc::new(EmptyExpander),
            }
        } else {
            sync::Arc::new(Expander(expander))
        };
    ProcMacro { name, kind, expander }
}

#[derive(Debug)]
struct Expander(proc_macro_api::ProcMacro);

impl ProcMacroExpander for Expander {
    fn expand(
        &self,
        subtree: &tt::Subtree,
        attrs: Option<&tt::Subtree>,
        env: &Env,
    ) -> Result<tt::Subtree, ProcMacroExpansionError> {
        let env = env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        match self.0.expand(subtree, attrs, env) {
            Ok(Ok(subtree)) => Ok(subtree),
            Ok(Err(err)) => Err(ProcMacroExpansionError::Panic(err.0)),
            Err(err) => Err(ProcMacroExpansionError::System(err.to_string())),
        }
    }
}

/// Dummy identity expander, used for attribute proc-macros that are deliberately ignored by the user.
#[derive(Debug)]
struct IdentityExpander;

impl ProcMacroExpander for IdentityExpander {
    fn expand(
        &self,
        subtree: &tt::Subtree,
        _: Option<&tt::Subtree>,
        _: &Env,
    ) -> Result<tt::Subtree, ProcMacroExpansionError> {
        Ok(subtree.clone())
    }
}

/// Empty expander, used for proc-macros that are deliberately ignored by the user.
#[derive(Debug)]
struct EmptyExpander;

impl ProcMacroExpander for EmptyExpander {
    fn expand(
        &self,
        _: &tt::Subtree,
        _: Option<&tt::Subtree>,
        _: &Env,
    ) -> Result<tt::Subtree, ProcMacroExpansionError> {
        Ok(tt::Subtree::empty())
    }
}

#[cfg(test)]
mod tests {
    use ide_db::base_db::SourceDatabase;

    use super::*;

    #[test]
    fn test_loading_rust_analyzer() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
        let cargo_config = CargoConfig::default();
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: false,
            with_proc_macro_server: ProcMacroServerChoice::None,
            prefill_caches: false,
        };
        let (host, _vfs, _proc_macro) =
            load_workspace_at(path, &cargo_config, &load_cargo_config, &|_| {}).unwrap();

        let n_crates = host.raw_database().crate_graph().iter().count();
        // RA has quite a few crates, but the exact count doesn't matter
        assert!(n_crates > 20);
    }
}
