//! The context or environment in which the language server functions.
//! In our server implementation this is know as the `WorldState`.
//!
//! Each tick provides an immutable snapshot of the state as `WorldSnapshot`.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crossbeam_channel::{unbounded, Receiver};
use lsp_server::ErrorCode;
use lsp_types::Url;
use parking_lot::RwLock;
use ra_cargo_watch::{
    url_from_path_with_drive_lowercasing, CheckOptions, CheckWatcher, CheckWatcherSharedState,
};
use ra_ide::{
    Analysis, AnalysisChange, AnalysisHost, CrateGraph, FeatureFlags, FileId, LibraryData,
    SourceRootId,
};
use ra_project_model::{get_rustc_cfg_options, ProjectWorkspace};
use ra_vfs::{LineEndings, RootEntry, Vfs, VfsChange, VfsFile, VfsRoot, VfsTask, Watch};
use ra_vfs_glob::{Glob, RustPackageFilterBuilder};
use relative_path::RelativePathBuf;

use crate::{
    main_loop::pending_requests::{CompletedRequest, LatestRequests},
    LspError, Result,
};

#[derive(Debug, Clone)]
pub struct Options {
    pub publish_decorations: bool,
    pub supports_location_link: bool,
    pub line_folding_only: bool,
    pub max_inlay_hint_length: Option<usize>,
    pub cargo_watch: CheckOptions,
}

/// `WorldState` is the primary mutable state of the language server
///
/// The most interesting components are `vfs`, which stores a consistent
/// snapshot of the file systems, and `analysis_host`, which stores our
/// incremental salsa database.
#[derive(Debug)]
pub struct WorldState {
    pub options: Options,
    //FIXME: this belongs to `LoopState` rather than to `WorldState`
    pub roots_to_scan: usize,
    pub roots: Vec<PathBuf>,
    pub workspaces: Arc<Vec<ProjectWorkspace>>,
    pub analysis_host: AnalysisHost,
    pub vfs: Arc<RwLock<Vfs>>,
    pub task_receiver: Receiver<VfsTask>,
    pub latest_requests: Arc<RwLock<LatestRequests>>,
    pub check_watcher: CheckWatcher,
}

/// An immutable snapshot of the world's state at a point in time.
pub struct WorldSnapshot {
    pub options: Options,
    pub workspaces: Arc<Vec<ProjectWorkspace>>,
    pub analysis: Analysis,
    pub vfs: Arc<RwLock<Vfs>>,
    pub latest_requests: Arc<RwLock<LatestRequests>>,
    pub check_watcher: Arc<RwLock<CheckWatcherSharedState>>,
}

impl WorldState {
    pub fn new(
        folder_roots: Vec<PathBuf>,
        workspaces: Vec<ProjectWorkspace>,
        lru_capacity: Option<usize>,
        exclude_globs: &[Glob],
        watch: Watch,
        options: Options,
        feature_flags: FeatureFlags,
    ) -> WorldState {
        let mut change = AnalysisChange::new();

        let mut roots = Vec::new();
        roots.extend(folder_roots.iter().map(|path| {
            let mut filter = RustPackageFilterBuilder::default().set_member(true);
            for glob in exclude_globs.iter() {
                filter = filter.exclude(glob.clone());
            }
            RootEntry::new(path.clone(), filter.into_vfs_filter())
        }));
        for ws in workspaces.iter() {
            roots.extend(ws.to_roots().into_iter().map(|pkg_root| {
                let mut filter =
                    RustPackageFilterBuilder::default().set_member(pkg_root.is_member());
                for glob in exclude_globs.iter() {
                    filter = filter.exclude(glob.clone());
                }
                RootEntry::new(pkg_root.path().clone(), filter.into_vfs_filter())
            }));
        }
        let (task_sender, task_receiver) = unbounded();
        let task_sender = Box::new(move |t| task_sender.send(t).unwrap());
        let (mut vfs, vfs_roots) = Vfs::new(roots, task_sender, watch);
        let roots_to_scan = vfs_roots.len();
        for r in vfs_roots {
            let vfs_root_path = vfs.root2path(r);
            let is_local = folder_roots.iter().any(|it| vfs_root_path.starts_with(it));
            change.add_root(SourceRootId(r.0), is_local);
            change.set_debug_root_path(SourceRootId(r.0), vfs_root_path.display().to_string());
        }

        // FIXME: Read default cfgs from config
        let default_cfg_options = {
            let mut opts = get_rustc_cfg_options();
            opts.insert_atom("test".into());
            opts.insert_atom("debug_assertion".into());
            opts
        };

        // Create crate graph from all the workspaces
        let mut crate_graph = CrateGraph::default();
        let mut load = |path: &std::path::Path| {
            let vfs_file = vfs.load(path);
            vfs_file.map(|f| FileId(f.0))
        };
        for ws in workspaces.iter() {
            let (graph, crate_names) = ws.to_crate_graph(&default_cfg_options, &mut load);
            let shift = crate_graph.extend(graph);
            for (crate_id, name) in crate_names {
                change.set_debug_crate_name(crate_id.shift(shift), name)
            }
        }
        change.set_crate_graph(crate_graph);

        // FIXME: Figure out the multi-workspace situation
        let check_watcher =
            CheckWatcher::new(&options.cargo_watch, folder_roots.first().cloned().unwrap());

        let mut analysis_host = AnalysisHost::new(lru_capacity, feature_flags);
        analysis_host.apply_change(change);
        WorldState {
            options,
            roots_to_scan,
            roots: folder_roots,
            workspaces: Arc::new(workspaces),
            analysis_host,
            vfs: Arc::new(RwLock::new(vfs)),
            task_receiver,
            latest_requests: Default::default(),
            check_watcher,
        }
    }

    /// Returns a vec of libraries
    /// FIXME: better API here
    pub fn process_changes(
        &mut self,
    ) -> Option<Vec<(SourceRootId, Vec<(FileId, RelativePathBuf, Arc<String>)>)>> {
        let changes = self.vfs.write().commit_changes();
        if changes.is_empty() {
            return None;
        }
        let mut libs = Vec::new();
        let mut change = AnalysisChange::new();
        for c in changes {
            match c {
                VfsChange::AddRoot { root, files } => {
                    let root_path = self.vfs.read().root2path(root);
                    let is_local = self.roots.iter().any(|r| root_path.starts_with(r));
                    if is_local {
                        self.roots_to_scan -= 1;
                        for (file, path, text) in files {
                            change.add_file(SourceRootId(root.0), FileId(file.0), path, text);
                        }
                    } else {
                        let files = files
                            .into_iter()
                            .map(|(vfsfile, path, text)| (FileId(vfsfile.0), path, text))
                            .collect();
                        libs.push((SourceRootId(root.0), files));
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
        Some(libs)
    }

    pub fn add_lib(&mut self, data: LibraryData) {
        self.roots_to_scan -= 1;
        let mut change = AnalysisChange::new();
        change.add_library(data);
        self.analysis_host.apply_change(change);
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        WorldSnapshot {
            options: self.options.clone(),
            workspaces: Arc::clone(&self.workspaces),
            analysis: self.analysis_host.analysis(),
            vfs: Arc::clone(&self.vfs),
            latest_requests: Arc::clone(&self.latest_requests),
            check_watcher: self.check_watcher.shared.clone(),
        }
    }

    pub fn maybe_collect_garbage(&mut self) {
        self.analysis_host.maybe_collect_garbage()
    }

    pub fn collect_garbage(&mut self) {
        self.analysis_host.collect_garbage()
    }

    pub fn complete_request(&mut self, request: CompletedRequest) {
        self.latest_requests.write().record(request)
    }

    pub fn feature_flags(&self) -> &FeatureFlags {
        self.analysis_host.feature_flags()
    }
}

impl WorldSnapshot {
    pub fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    pub fn uri_to_file_id(&self, uri: &Url) -> Result<FileId> {
        let path = uri.to_file_path().map_err(|()| format!("invalid uri: {}", uri))?;
        let file = self.vfs.read().path2file(&path).ok_or_else(|| {
            // Show warning as this file is outside current workspace
            LspError {
                code: ErrorCode::InvalidRequest as i32,
                message: "Rust file outside current workspace is not supported yet.".to_string(),
            }
        })?;
        Ok(FileId(file.0))
    }

    pub fn file_id_to_uri(&self, id: FileId) -> Result<Url> {
        let path = self.vfs.read().file2path(VfsFile(id.0));
        let url = url_from_path_with_drive_lowercasing(path)?;

        Ok(url)
    }

    pub fn file_line_endings(&self, id: FileId) -> LineEndings {
        self.vfs.read().file_line_endings(VfsFile(id.0))
    }

    pub fn path_to_uri(&self, root: SourceRootId, path: &RelativePathBuf) -> Result<Url> {
        let base = self.vfs.read().root2path(VfsRoot(root.0));
        let path = path.to_path(base);
        let url = Url::from_file_path(&path)
            .map_err(|_| format!("can't convert path to url: {}", path.display()))?;
        Ok(url)
    }

    pub fn status(&self) -> String {
        let mut res = String::new();
        if self.workspaces.is_empty() {
            res.push_str("no workspaces\n")
        } else {
            res.push_str("workspaces:\n");
            for w in self.workspaces.iter() {
                res += &format!("{} packages loaded\n", w.n_packages());
            }
        }
        res.push_str("\nanalysis:\n");
        res.push_str(
            &self
                .analysis
                .status()
                .unwrap_or_else(|_| "Analysis retrieval was cancelled".to_owned()),
        );
        res
    }

    pub fn workspace_root_for(&self, file_id: FileId) -> Option<&Path> {
        let path = self.vfs.read().file2path(VfsFile(file_id.0));
        self.workspaces.iter().find_map(|ws| ws.workspace_root_for(&path))
    }

    pub fn feature_flags(&self) -> &FeatureFlags {
        self.analysis.feature_flags()
    }
}
