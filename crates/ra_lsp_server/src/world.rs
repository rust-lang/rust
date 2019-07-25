use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use gen_lsp_server::ErrorCode;
use lsp_types::Url;
use parking_lot::RwLock;
use ra_ide_api::{
    Analysis, AnalysisChange, AnalysisHost, CrateGraph, FileId, LibraryData, SourceRootId,
};
use ra_vfs::{Vfs, VfsChange, VfsFile, VfsRoot};
use relative_path::RelativePathBuf;

use crate::{
    main_loop::pending_requests::{CompletedRequest, LatestRequests},
    project_model::ProjectWorkspace,
    vfs_filter::IncludeRustFiles,
    LspError, Result,
};

#[derive(Debug, Clone)]
pub struct Options {
    pub publish_decorations: bool,
    pub show_workspace_loaded: bool,
    pub supports_location_link: bool,
}

/// `WorldState` is the primary mutable state of the language server
///
/// The most interesting components are `vfs`, which stores a consistent
/// snapshot of the file systems, and `analysis_host`, which stores our
/// incremental salsa database.
#[derive(Debug)]
pub struct WorldState {
    pub options: Options,
    pub roots_to_scan: usize,
    pub roots: Vec<PathBuf>,
    pub workspaces: Arc<Vec<ProjectWorkspace>>,
    pub analysis_host: AnalysisHost,
    pub vfs: Arc<RwLock<Vfs>>,
    pub latest_requests: Arc<RwLock<LatestRequests>>,
}

/// An immutable snapshot of the world's state at a point in time.
pub struct WorldSnapshot {
    pub options: Options,
    pub workspaces: Arc<Vec<ProjectWorkspace>>,
    pub analysis: Analysis,
    pub vfs: Arc<RwLock<Vfs>>,
    pub latest_requests: Arc<RwLock<LatestRequests>>,
}

impl WorldState {
    pub fn new(
        folder_roots: Vec<PathBuf>,
        workspaces: Vec<ProjectWorkspace>,
        lru_capacity: Option<usize>,
        options: Options,
    ) -> WorldState {
        let mut change = AnalysisChange::new();

        let mut roots = Vec::new();
        roots.extend(folder_roots.iter().cloned().map(IncludeRustFiles::member));
        for ws in workspaces.iter() {
            roots.extend(IncludeRustFiles::from_roots(ws.to_roots()));
        }

        let (mut vfs, vfs_roots) = Vfs::new(roots);
        let roots_to_scan = vfs_roots.len();
        for r in vfs_roots {
            let vfs_root_path = vfs.root2path(r);
            let is_local = folder_roots.iter().any(|it| vfs_root_path.starts_with(it));
            change.add_root(SourceRootId(r.0), is_local);
        }

        // Create crate graph from all the workspaces
        let mut crate_graph = CrateGraph::default();
        let mut load = |path: &std::path::Path| {
            let vfs_file = vfs.load(path);
            vfs_file.map(|f| FileId(f.0))
        };
        for ws in workspaces.iter() {
            crate_graph.extend(ws.to_crate_graph(&mut load));
        }
        change.set_crate_graph(crate_graph);

        let mut analysis_host = AnalysisHost::new(lru_capacity);
        analysis_host.apply_change(change);
        WorldState {
            options,
            roots_to_scan,
            roots: folder_roots,
            workspaces: Arc::new(workspaces),
            analysis_host,
            vfs: Arc::new(RwLock::new(vfs)),
            latest_requests: Default::default(),
        }
    }

    /// Returns a vec of libraries
    /// FIXME: better API here
    pub fn process_changes(
        &mut self,
    ) -> Vec<(SourceRootId, Vec<(FileId, RelativePathBuf, Arc<String>)>)> {
        let changes = self.vfs.write().commit_changes();
        if changes.is_empty() {
            return Vec::new();
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
        libs
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
        let url = Url::from_file_path(&path)
            .map_err(|_| format!("can't convert path to url: {}", path.display()))?;
        Ok(url)
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
                res += &format!("{} packages loaded\n", w.count());
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
}
