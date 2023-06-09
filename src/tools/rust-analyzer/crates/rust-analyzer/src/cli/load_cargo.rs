//! Loads a Cargo project into a static instance of analysis, without support
//! for incorporating changes.
use std::path::Path;

use anyhow::{anyhow, Result};
use crossbeam_channel::{unbounded, Receiver};
use ide::{AnalysisHost, Change};
use ide_db::{
    base_db::{CrateGraph, ProcMacros},
    FxHashMap,
};
use proc_macro_api::ProcMacroServer;
use project_model::{CargoConfig, ProjectManifest, ProjectWorkspace};
use triomphe::Arc;
use vfs::{loader::Handle, AbsPath, AbsPathBuf};

use crate::reload::{load_proc_macro, ProjectFolders, SourceRootConfig};

// Note: Since this type is used by external tools that use rust-analyzer as a library
// what otherwise would be `pub(crate)` has to be `pub` here instead.
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

// Note: Since this function is used by external tools that use rust-analyzer as a library
// what otherwise would be `pub(crate)` has to be `pub` here instead.
pub fn load_workspace_at(
    root: &Path,
    cargo_config: &CargoConfig,
    load_config: &LoadCargoConfig,
    progress: &dyn Fn(String),
) -> Result<(AnalysisHost, vfs::Vfs, Option<ProcMacroServer>)> {
    let root = AbsPathBuf::assert(std::env::current_dir()?.join(root));
    let root = ProjectManifest::discover_single(&root)?;
    let mut workspace = ProjectWorkspace::load(root, cargo_config, progress)?;

    if load_config.load_out_dirs_from_check {
        let build_scripts = workspace.run_build_scripts(cargo_config, progress)?;
        workspace.set_build_scripts(build_scripts)
    }

    load_workspace(workspace, &cargo_config.extra_env, load_config)
}

// Note: Since this function is used by external tools that use rust-analyzer as a library
// what otherwise would be `pub(crate)` has to be `pub` here instead.
//
// The reason both, `load_workspace_at` and `load_workspace` are `pub` is that some of
// these tools need access to `ProjectWorkspace`, too, which `load_workspace_at` hides.
pub fn load_workspace(
    ws: ProjectWorkspace,
    extra_env: &FxHashMap<String, String>,
    load_config: &LoadCargoConfig,
) -> Result<(AnalysisHost, vfs::Vfs, Option<ProcMacroServer>)> {
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
        ProcMacroServerChoice::None => Err(anyhow!("proc macro server disabled")),
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

    tracing::debug!("crate graph: {:?}", crate_graph);
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
                analysis_change.change_file(file.file_id, Some(Arc::from(text)))
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

#[cfg(test)]
mod tests {
    use super::*;

    use hir::Crate;

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

        let n_crates = Crate::all(host.raw_database()).len();
        // RA has quite a few crates, but the exact count doesn't matter
        assert!(n_crates > 20);
    }
}
