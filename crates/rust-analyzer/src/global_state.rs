//! The context or environment in which the language server functions. In our
//! server implementation this is know as the `WorldState`.
//!
//! Each tick provides an immutable snapshot of the state as `WorldSnapshot`.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crossbeam_channel::{unbounded, Receiver};
use lsp_types::Url;
use parking_lot::RwLock;
use ra_flycheck::{Flycheck, FlycheckConfig};
use ra_ide::{Analysis, AnalysisChange, AnalysisHost, CrateGraph, FileId, SourceRootId};
use ra_project_model::{CargoWorkspace, ProcMacroClient, ProjectWorkspace, Target};
use ra_vfs::{LineEndings, RootEntry, Vfs, VfsChange, VfsFile, VfsTask, Watch};
use stdx::format_to;

use crate::{
    config::{Config, FilesWatcher},
    diagnostics::{CheckFixes, DiagnosticCollection},
    main_loop::req_queue::{CompletedInRequest, LatestRequests},
    to_proto::url_from_abs_path,
    vfs_glob::{Glob, RustPackageFilterBuilder},
    LspError, Result,
};
use ra_db::{CrateId, ExternSourceId};
use rustc_hash::{FxHashMap, FxHashSet};

fn create_flycheck(workspaces: &[ProjectWorkspace], config: &FlycheckConfig) -> Option<Flycheck> {
    // FIXME: Figure out the multi-workspace situation
    workspaces.iter().find_map(|w| match w {
        ProjectWorkspace::Cargo { cargo, .. } => {
            let cargo_project_root = cargo.workspace_root().to_path_buf();
            Some(Flycheck::new(config.clone(), cargo_project_root))
        }
        ProjectWorkspace::Json { .. } => {
            log::warn!("Cargo check watching only supported for cargo workspaces, disabling");
            None
        }
    })
}

/// `GlobalState` is the primary mutable state of the language server
///
/// The most interesting components are `vfs`, which stores a consistent
/// snapshot of the file systems, and `analysis_host`, which stores our
/// incremental salsa database.
#[derive(Debug)]
pub struct GlobalState {
    pub config: Config,
    pub local_roots: Vec<PathBuf>,
    pub workspaces: Arc<Vec<ProjectWorkspace>>,
    pub analysis_host: AnalysisHost,
    pub vfs: Arc<RwLock<Vfs>>,
    pub task_receiver: Receiver<VfsTask>,
    pub latest_requests: Arc<RwLock<LatestRequests>>,
    pub flycheck: Option<Flycheck>,
    pub diagnostics: DiagnosticCollection,
    pub proc_macro_client: ProcMacroClient,
}

/// An immutable snapshot of the world's state at a point in time.
pub struct GlobalStateSnapshot {
    pub config: Config,
    pub workspaces: Arc<Vec<ProjectWorkspace>>,
    pub analysis: Analysis,
    pub latest_requests: Arc<RwLock<LatestRequests>>,
    pub check_fixes: CheckFixes,
    vfs: Arc<RwLock<Vfs>>,
}

impl GlobalState {
    pub fn new(
        workspaces: Vec<ProjectWorkspace>,
        lru_capacity: Option<usize>,
        exclude_globs: &[Glob],
        config: Config,
    ) -> GlobalState {
        let mut change = AnalysisChange::new();

        let mut extern_dirs: FxHashSet<PathBuf> = FxHashSet::default();

        let mut local_roots = Vec::new();
        let roots: Vec<_> = {
            let create_filter = |is_member| {
                RustPackageFilterBuilder::default()
                    .set_member(is_member)
                    .exclude(exclude_globs.iter().cloned())
                    .into_vfs_filter()
            };
            let mut roots = Vec::new();
            for root in workspaces.iter().flat_map(ProjectWorkspace::to_roots) {
                let path = root.path().to_owned();
                if root.is_member() {
                    local_roots.push(path.clone());
                }
                roots.push(RootEntry::new(path, create_filter(root.is_member())));
                if let Some(out_dir) = root.out_dir() {
                    extern_dirs.insert(out_dir.to_path_buf());
                    roots.push(RootEntry::new(
                        out_dir.to_path_buf(),
                        create_filter(root.is_member()),
                    ))
                }
            }
            roots
        };

        let (task_sender, task_receiver) = unbounded();
        let task_sender = Box::new(move |t| task_sender.send(t).unwrap());
        let watch = Watch(matches!(config.files.watcher, FilesWatcher::Notify));
        let (mut vfs, vfs_roots) = Vfs::new(roots, task_sender, watch);

        let mut extern_source_roots = FxHashMap::default();
        for r in vfs_roots {
            let vfs_root_path = vfs.root2path(r);
            let is_local = local_roots.iter().any(|it| vfs_root_path.starts_with(it));
            change.add_root(SourceRootId(r.0), is_local);

            // FIXME: add path2root in vfs to simpily this logic
            if extern_dirs.contains(&vfs_root_path) {
                extern_source_roots.insert(vfs_root_path, ExternSourceId(r.0));
            }
        }

        let proc_macro_client = match &config.proc_macro_srv {
            None => ProcMacroClient::dummy(),
            Some((path, args)) => match ProcMacroClient::extern_process(path.into(), args) {
                Ok(it) => it,
                Err(err) => {
                    log::error!(
                        "Failed to run ra_proc_macro_srv from path {}, error: {:?}",
                        path.display(),
                        err
                    );
                    ProcMacroClient::dummy()
                }
            },
        };

        // Create crate graph from all the workspaces
        let mut crate_graph = CrateGraph::default();
        let mut load = |path: &Path| {
            // Some path from metadata will be non canonicalized, e.g. /foo/../bar/lib.rs
            let path = path.canonicalize().ok()?;
            let vfs_file = vfs.load(&path);
            vfs_file.map(|f| FileId(f.0))
        };
        for ws in workspaces.iter() {
            crate_graph.extend(ws.to_crate_graph(
                config.cargo.target.as_deref(),
                &extern_source_roots,
                &proc_macro_client,
                &mut load,
            ));
        }
        change.set_crate_graph(crate_graph);

        let flycheck = config.check.as_ref().and_then(|c| create_flycheck(&workspaces, c));

        let mut analysis_host = AnalysisHost::new(lru_capacity);
        analysis_host.apply_change(change);
        GlobalState {
            config,
            local_roots,
            workspaces: Arc::new(workspaces),
            analysis_host,
            vfs: Arc::new(RwLock::new(vfs)),
            task_receiver,
            latest_requests: Default::default(),
            flycheck,
            diagnostics: Default::default(),
            proc_macro_client,
        }
    }

    pub fn update_configuration(&mut self, config: Config) {
        self.analysis_host.update_lru_capacity(config.lru_capacity);
        if config.check != self.config.check {
            self.flycheck =
                config.check.as_ref().and_then(|it| create_flycheck(&self.workspaces, it));
        }

        self.config = config;
    }

    /// Returns a vec of libraries
    /// FIXME: better API here
    pub fn process_changes(&mut self, roots_scanned: &mut usize) -> bool {
        let changes = self.vfs.write().commit_changes();
        if changes.is_empty() {
            return false;
        }
        let mut change = AnalysisChange::new();
        for c in changes {
            match c {
                VfsChange::AddRoot { root, files } => {
                    *roots_scanned += 1;
                    for (file, path, text) in files {
                        change.add_file(SourceRootId(root.0), FileId(file.0), path, text);
                    }
                }
                VfsChange::AddFile { root, file, path, text } => {
                    change.add_file(SourceRootId(root.0), FileId(file.0), path, text);
                }
                VfsChange::RemoveFile { root, file, path } => {
                    change.remove_file(SourceRootId(root.0), FileId(file.0), path)
                }
                VfsChange::ChangeFile { file, text } => {
                    change.change_file(FileId(file.0), text);
                }
            }
        }
        self.analysis_host.apply_change(change);
        true
    }

    pub fn snapshot(&self) -> GlobalStateSnapshot {
        GlobalStateSnapshot {
            config: self.config.clone(),
            workspaces: Arc::clone(&self.workspaces),
            analysis: self.analysis_host.analysis(),
            vfs: Arc::clone(&self.vfs),
            latest_requests: Arc::clone(&self.latest_requests),
            check_fixes: Arc::clone(&self.diagnostics.check_fixes),
        }
    }

    pub fn maybe_collect_garbage(&mut self) {
        self.analysis_host.maybe_collect_garbage()
    }

    pub fn collect_garbage(&mut self) {
        self.analysis_host.collect_garbage()
    }

    pub(crate) fn complete_request(&mut self, request: CompletedInRequest) {
        self.latest_requests.write().record(request)
    }
}

impl GlobalStateSnapshot {
    pub fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    pub fn url_to_file_id(&self, url: &Url) -> Result<FileId> {
        let path = url.to_file_path().map_err(|()| format!("invalid uri: {}", url))?;
        let file = self.vfs.read().path2file(&path).ok_or_else(|| {
            // Show warning as this file is outside current workspace
            // FIXME: just handle such files, and remove `LspError::UNKNOWN_FILE`.
            LspError {
                code: LspError::UNKNOWN_FILE,
                message: "Rust file outside current workspace is not supported yet.".to_string(),
            }
        })?;
        Ok(FileId(file.0))
    }

    pub fn file_id_to_url(&self, id: FileId) -> Url {
        file_id_to_url(&self.vfs.read(), id)
    }

    pub fn file_line_endings(&self, id: FileId) -> LineEndings {
        self.vfs.read().file_line_endings(VfsFile(id.0))
    }

    pub fn anchored_path(&self, file_id: FileId, path: &str) -> Url {
        let mut base = self.vfs.read().file2path(VfsFile(file_id.0));
        base.pop();
        let path = base.join(path);
        url_from_abs_path(&path)
    }

    pub(crate) fn cargo_target_for_crate_root(
        &self,
        crate_id: CrateId,
    ) -> Option<(&CargoWorkspace, Target)> {
        let file_id = self.analysis().crate_root(crate_id).ok()?;
        let path = self.vfs.read().file2path(VfsFile(file_id.0));
        self.workspaces.iter().find_map(|ws| match ws {
            ProjectWorkspace::Cargo { cargo, .. } => {
                cargo.target_by_root(&path).map(|it| (cargo, it))
            }
            ProjectWorkspace::Json { .. } => None,
        })
    }

    pub fn status(&self) -> String {
        let mut buf = String::new();
        if self.workspaces.is_empty() {
            buf.push_str("no workspaces\n")
        } else {
            buf.push_str("workspaces:\n");
            for w in self.workspaces.iter() {
                format_to!(buf, "{} packages loaded\n", w.n_packages());
            }
        }
        buf.push_str("\nanalysis:\n");
        buf.push_str(
            &self
                .analysis
                .status()
                .unwrap_or_else(|_| "Analysis retrieval was cancelled".to_owned()),
        );
        buf
    }

    pub fn workspace_root_for(&self, file_id: FileId) -> Option<&Path> {
        let path = self.vfs.read().file2path(VfsFile(file_id.0));
        self.workspaces.iter().find_map(|ws| ws.workspace_root_for(&path))
    }
}

pub(crate) fn file_id_to_url(vfs: &Vfs, id: FileId) -> Url {
    let path = vfs.file2path(VfsFile(id.0));
    url_from_abs_path(&path)
}
