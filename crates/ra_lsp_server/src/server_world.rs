use std::{
    path::{PathBuf},
    sync::Arc,
};

use languageserver_types::Url;
use ra_analysis::{
    Analysis, AnalysisChange, AnalysisHost, CrateGraph, FileId, LibraryData,
    SourceRootId
};
use ra_vfs::{Vfs, VfsChange, VfsFile};
use rustc_hash::FxHashMap;
use relative_path::RelativePathBuf;
use parking_lot::RwLock;
use failure::{format_err};

use crate::{
    project_model::{CargoWorkspace, TargetKind},
    Result,
};

#[derive(Debug)]
pub struct ServerWorldState {
    pub roots_to_scan: usize,
    pub root: PathBuf,
    pub workspaces: Arc<Vec<CargoWorkspace>>,
    pub analysis_host: AnalysisHost,
    pub vfs: Arc<RwLock<Vfs>>,
}

pub struct ServerWorld {
    pub workspaces: Arc<Vec<CargoWorkspace>>,
    pub analysis: Analysis,
    pub vfs: Arc<RwLock<Vfs>>,
}

impl ServerWorldState {
    pub fn new(root: PathBuf, workspaces: Vec<CargoWorkspace>) -> ServerWorldState {
        let mut change = AnalysisChange::new();

        let mut roots = Vec::new();
        roots.push(root.clone());
        for ws in workspaces.iter() {
            for pkg in ws.packages() {
                roots.push(pkg.root(&ws).to_path_buf());
            }
        }
        let roots_to_scan = roots.len();
        let (mut vfs, roots) = Vfs::new(roots);
        for r in roots {
            change.add_root(SourceRootId(r.0));
        }

        let mut crate_graph = CrateGraph::default();
        let mut pkg_to_lib_crate = FxHashMap::default();
        let mut pkg_crates = FxHashMap::default();
        for ws in workspaces.iter() {
            for pkg in ws.packages() {
                for tgt in pkg.targets(ws) {
                    let root = tgt.root(ws);
                    if let Some(file_id) = vfs.load(root) {
                        let file_id = FileId(file_id.0);
                        let crate_id = crate_graph.add_crate_root(file_id);
                        if tgt.kind(ws) == TargetKind::Lib {
                            pkg_to_lib_crate.insert(pkg, crate_id);
                        }
                        pkg_crates
                            .entry(pkg)
                            .or_insert_with(Vec::new)
                            .push(crate_id);
                    }
                }
            }
            for pkg in ws.packages() {
                for dep in pkg.dependencies(ws) {
                    if let Some(&to) = pkg_to_lib_crate.get(&dep.pkg) {
                        for &from in pkg_crates.get(&pkg).into_iter().flatten() {
                            crate_graph.add_dep(from, dep.name.clone(), to);
                        }
                    }
                }
            }
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
            log::info!("vfs change {:?}", c);
            match c {
                VfsChange::AddRoot { root, files } => {
                    let root_path = self.vfs.read().root2path(root);
                    if root_path.starts_with(&self.root) {
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
                VfsChange::AddFile {
                    root,
                    file,
                    path,
                    text,
                } => {
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

    pub fn snapshot(&self) -> ServerWorld {
        ServerWorld {
            workspaces: Arc::clone(&self.workspaces),
            analysis: self.analysis_host.analysis(),
            vfs: Arc::clone(&self.vfs),
        }
    }
}

impl ServerWorld {
    pub fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    pub fn uri_to_file_id(&self, uri: &Url) -> Result<FileId> {
        let path = uri
            .to_file_path()
            .map_err(|()| format_err!("invalid uri: {}", uri))?;
        let file = self
            .vfs
            .read()
            .path2file(&path)
            .ok_or_else(|| format_err!("unknown file: {}", path.display()))?;
        Ok(FileId(file.0))
    }

    pub fn file_id_to_uri(&self, id: FileId) -> Result<Url> {
        let path = self.vfs.read().file2path(VfsFile(id.0));
        let url = Url::from_file_path(&path)
            .map_err(|_| format_err!("can't convert path to url: {}", path.display()))?;
        Ok(url)
    }
}
