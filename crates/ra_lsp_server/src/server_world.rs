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
use rustc_hash::FxHashMap;
use relative_path::RelativePathBuf;
use parking_lot::RwLock;
use failure::format_err;

use crate::{
    project_model::{ProjectWorkspace, TargetKind},
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
            for pkg in ws.cargo.packages() {
                roots.push(pkg.root(&ws.cargo).to_path_buf());
            }
            for krate in ws.sysroot.crates() {
                roots.push(krate.root_dir(&ws.sysroot).to_path_buf())
            }
        }
        roots.sort();
        roots.dedup();
        let roots_to_scan = roots.len();
        let (mut vfs, roots) = Vfs::new(roots);
        for r in roots {
            let is_local = vfs.root2path(r).starts_with(&root);
            change.add_root(SourceRootId(r.0.into()), is_local);
        }

        let mut crate_graph = CrateGraph::default();
        for ws in workspaces.iter() {
            // First, load std
            let mut sysroot_crates = FxHashMap::default();
            for krate in ws.sysroot.crates() {
                if let Some(file_id) = vfs.load(krate.root(&ws.sysroot)) {
                    let file_id = FileId(file_id.0.into());
                    sysroot_crates.insert(krate, crate_graph.add_crate_root(file_id));
                }
            }
            for from in ws.sysroot.crates() {
                for to in from.deps(&ws.sysroot) {
                    let name = to.name(&ws.sysroot);
                    if let (Some(&from), Some(&to)) =
                        (sysroot_crates.get(&from), sysroot_crates.get(&to))
                    {
                        if let Err(_) = crate_graph.add_dep(from, name.clone(), to) {
                            log::error!("cyclic dependency between sysroot crates")
                        }
                    }
                }
            }

            let libstd = ws
                .sysroot
                .std()
                .and_then(|it| sysroot_crates.get(&it).map(|&it| it));

            let mut pkg_to_lib_crate = FxHashMap::default();
            let mut pkg_crates = FxHashMap::default();
            // Next, create crates for each package, target pair
            for pkg in ws.cargo.packages() {
                let mut lib_tgt = None;
                for tgt in pkg.targets(&ws.cargo) {
                    let root = tgt.root(&ws.cargo);
                    if let Some(file_id) = vfs.load(root) {
                        let file_id = FileId(file_id.0.into());
                        let crate_id = crate_graph.add_crate_root(file_id);
                        if tgt.kind(&ws.cargo) == TargetKind::Lib {
                            lib_tgt = Some(crate_id);
                            pkg_to_lib_crate.insert(pkg, crate_id);
                        }
                        pkg_crates
                            .entry(pkg)
                            .or_insert_with(Vec::new)
                            .push(crate_id);
                    }
                }

                // Set deps to the std and to the lib target of the current package
                for &from in pkg_crates.get(&pkg).into_iter().flatten() {
                    if let Some(to) = lib_tgt {
                        if to != from {
                            if let Err(_) =
                                crate_graph.add_dep(from, pkg.name(&ws.cargo).into(), to)
                            {
                                log::error!(
                                    "cyclic dependency between targets of {}",
                                    pkg.name(&ws.cargo)
                                )
                            }
                        }
                    }
                    if let Some(std) = libstd {
                        if let Err(_) = crate_graph.add_dep(from, "std".into(), std) {
                            log::error!("cyclic dependency on std for {}", pkg.name(&ws.cargo))
                        }
                    }
                }
            }

            // Now add a dep ednge from all targets of upstream to the lib
            // target of downstream.
            for pkg in ws.cargo.packages() {
                for dep in pkg.dependencies(&ws.cargo) {
                    if let Some(&to) = pkg_to_lib_crate.get(&dep.pkg) {
                        for &from in pkg_crates.get(&pkg).into_iter().flatten() {
                            if let Err(_) = crate_graph.add_dep(from, dep.name.clone(), to) {
                                log::error!(
                                    "cyclic dependency {} -> {}",
                                    pkg.name(&ws.cargo),
                                    dep.pkg.name(&ws.cargo)
                                )
                            }
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
                VfsChange::AddFile {
                    root,
                    file,
                    path,
                    text,
                } => {
                    change.add_file(
                        SourceRootId(root.0.into()),
                        FileId(file.0.into()),
                        path,
                        text,
                    );
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
                res += &format!("{} packages loaded\n", w.cargo.packages().count());
            }
        }
        res.push_str("\nanalysis:\n");
        res.push_str(&self.analysis.status());
        res
    }
}
