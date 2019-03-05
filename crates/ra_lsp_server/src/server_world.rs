use std::{
    path::PathBuf,
    sync::Arc,
};

use lsp_types::Url;
use ra_ide_api::{
    Analysis, AnalysisChange, AnalysisHost, CrateGraph, FileId, LibraryData,
    SourceRootId
};
use ra_vfs::{Vfs, VfsChange, VfsFile, VfsRoot};
use relative_path::RelativePathBuf;
use parking_lot::RwLock;
use failure::format_err;

use crate::{
    project_model::ProjectWorkspace,
    Result,
};

#[derive(Debug)]
pub struct ServerWorldState {
    pub roots_to_scan: usize,
    pub root: PathBuf,
    pub workspaces: Arc<Vec<ProjectWorkspace>>,
    pub analysis_host: AnalysisHost,
    pub vfs: Arc<RwLock<Vfs>>,
}

pub struct ServerWorld {
    pub workspaces: Arc<Vec<ProjectWorkspace>>,
    pub analysis: Analysis,
    pub vfs: Arc<RwLock<Vfs>>,
}

impl ServerWorldState {
    pub fn new(root: PathBuf, workspaces: Vec<ProjectWorkspace>) -> ServerWorldState {
        let mut change = AnalysisChange::new();

        let mut roots = Vec::new();
        roots.push(root.clone());
        for ws in workspaces.iter() {
            ws.add_roots(&mut roots);
        }
        let (mut vfs, roots) = Vfs::new(roots);
        let roots_to_scan = roots.len();
        for r in roots {
            let is_local = vfs.root2path(r).starts_with(&root);
            change.add_root(SourceRootId(r.0.into()), is_local);
        }

        // Create crate graph from all the workspaces
        let mut crate_graph = CrateGraph::default();
        let mut load = |path: &std::path::Path| {
            let vfs_file = vfs.load(path);
            vfs_file.map(|f| FileId(f.0.into()))
        };
        for ws in workspaces.iter() {
            crate_graph.extend(ws.to_crate_graph(&mut load));
        }
        change.set_crate_graph(crate_graph);

        let mut analysis_host = AnalysisHost::default();
        analysis_host.apply_change(change);
        ServerWorldState {
            roots_to_scan,
            root,
            workspaces: Arc::new(workspaces),
            analysis_host,
            vfs: Arc::new(RwLock::new(vfs)),
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
                    if root_path.starts_with(&self.root) {
                        self.roots_to_scan -= 1;
                        for (file, path, text) in files {
                            change.add_file(
                                SourceRootId(root.0.into()),
                                FileId(file.0.into()),
                                path,
                                text,
                            );
                        }
                    } else {
                        let files = files
                            .into_iter()
                            .map(|(vfsfile, path, text)| (FileId(vfsfile.0.into()), path, text))
                            .collect();
                        libs.push((SourceRootId(root.0.into()), files));
                    }
                }
                VfsChange::AddFile { root, file, path, text } => {
                    change.add_file(SourceRootId(root.0.into()), FileId(file.0.into()), path, text);
                }
                VfsChange::RemoveFile { root, file, path } => {
                    change.remove_file(SourceRootId(root.0.into()), FileId(file.0.into()), path)
                }
                VfsChange::ChangeFile { file, text } => {
                    change.change_file(FileId(file.0.into()), text);
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

    pub fn snapshot(&self) -> ServerWorld {
        ServerWorld {
            workspaces: Arc::clone(&self.workspaces),
            analysis: self.analysis_host.analysis(),
            vfs: Arc::clone(&self.vfs),
        }
    }

    pub fn maybe_collect_garbage(&mut self) {
        self.analysis_host.maybe_collect_garbage()
    }

    pub fn collect_garbage(&mut self) {
        self.analysis_host.collect_garbage()
    }
}

impl ServerWorld {
    pub fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    pub fn uri_to_file_id(&self, uri: &Url) -> Result<FileId> {
        let path = uri.to_file_path().map_err(|()| format_err!("invalid uri: {}", uri))?;
        let file = self
            .vfs
            .read()
            .path2file(&path)
            .ok_or_else(|| format_err!("unknown file: {}", path.display()))?;
        Ok(FileId(file.0.into()))
    }

    pub fn file_id_to_uri(&self, id: FileId) -> Result<Url> {
        let path = self.vfs.read().file2path(VfsFile(id.0.into()));
        let url = Url::from_file_path(&path)
            .map_err(|_| format_err!("can't convert path to url: {}", path.display()))?;
        Ok(url)
    }

    pub fn path_to_uri(&self, root: SourceRootId, path: &RelativePathBuf) -> Result<Url> {
        let base = self.vfs.read().root2path(VfsRoot(root.0.into()));
        let path = path.to_path(base);
        let url = Url::from_file_path(&path)
            .map_err(|_| format_err!("can't convert path to url: {}", path.display()))?;
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
        res.push_str(&self.analysis.status());
        res
    }
}
