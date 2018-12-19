use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use languageserver_types::Url;
use ra_analysis::{
    Analysis, AnalysisChange, AnalysisHost, CrateGraph, FileId, LibraryData,
};
use rustc_hash::FxHashMap;
use failure::{bail, format_err};

use crate::{
    path_map::{PathMap, Root},
    project_model::{CargoWorkspace, TargetKind},
    vfs::{FileEvent, FileEventKind},
    Result,
};

#[derive(Debug, Default)]
pub struct ServerWorldState {
    pub workspaces: Arc<Vec<CargoWorkspace>>,
    pub analysis_host: AnalysisHost,
    pub path_map: PathMap,
    pub mem_map: FxHashMap<FileId, Option<String>>,
}

pub struct ServerWorld {
    pub workspaces: Arc<Vec<CargoWorkspace>>,
    pub analysis: Analysis,
    pub path_map: PathMap,
}

impl ServerWorldState {
    pub fn apply_fs_changes(&mut self, events: Vec<FileEvent>) {
        let mut change = AnalysisChange::new();
        let mut inserted = false;
        {
            let pm = &mut self.path_map;
            let mm = &mut self.mem_map;
            events
                .into_iter()
                .map(|event| {
                    let text = match event.kind {
                        FileEventKind::Add(text) => text,
                    };
                    (event.path, text)
                })
                .map(|(path, text)| {
                    let (ins, file_id) = pm.get_or_insert(path, Root::Workspace);
                    inserted |= ins;
                    (file_id, text)
                })
                .filter_map(|(file_id, text)| {
                    if mm.contains_key(&file_id) {
                        mm.insert(file_id, Some(text));
                        None
                    } else {
                        Some((file_id, text))
                    }
                })
                .for_each(|(file_id, text)| change.add_file(file_id, text));
        }
        if inserted {
            change.set_file_resolver(Arc::new(self.path_map.clone()))
        }
        self.analysis_host.apply_change(change);
    }
    pub fn events_to_files(
        &mut self,
        events: Vec<FileEvent>,
    ) -> (Vec<(FileId, String)>, Arc<FileResolver>) {
        let files = {
            let pm = &mut self.path_map;
            events
                .into_iter()
                .map(|event| {
                    let FileEventKind::Add(text) = event.kind;
                    (event.path, text)
                })
                .map(|(path, text)| (pm.get_or_insert(path, Root::Lib).1, text))
                .collect()
        };
        let resolver = Arc::new(self.path_map.clone());
        (files, resolver)
    }
    pub fn add_lib(&mut self, data: LibraryData) {
        let mut change = AnalysisChange::new();
        change.add_library(data);
        self.analysis_host.apply_change(change);
    }

    pub fn add_mem_file(&mut self, path: PathBuf, text: String) -> FileId {
        let (inserted, file_id) = self.path_map.get_or_insert(path, Root::Workspace);
        if self.path_map.get_root(file_id) != Root::Lib {
            let mut change = AnalysisChange::new();
            if inserted {
                change.add_file(file_id, text);
                change.set_file_resolver(Arc::new(self.path_map.clone()));
            } else {
                change.change_file(file_id, text);
            }
            self.analysis_host.apply_change(change);
        }
        self.mem_map.insert(file_id, None);
        file_id
    }

    pub fn change_mem_file(&mut self, path: &Path, text: String) -> Result<()> {
        let file_id = self
            .path_map
            .get_id(path)
            .ok_or_else(|| format_err!("change to unknown file: {}", path.display()))?;
        if self.path_map.get_root(file_id) != Root::Lib {
            let mut change = AnalysisChange::new();
            change.change_file(file_id, text);
            self.analysis_host.apply_change(change);
        }
        Ok(())
    }

    pub fn remove_mem_file(&mut self, path: &Path) -> Result<FileId> {
        let file_id = self
            .path_map
            .get_id(path)
            .ok_or_else(|| format_err!("change to unknown file: {}", path.display()))?;
        match self.mem_map.remove(&file_id) {
            Some(_) => (),
            None => bail!("unmatched close notification"),
        };
        // Do this via file watcher ideally.
        let text = fs::read_to_string(path).ok();
        if self.path_map.get_root(file_id) != Root::Lib {
            let mut change = AnalysisChange::new();
            if let Some(text) = text {
                change.change_file(file_id, text);
            }
            self.analysis_host.apply_change(change);
        }
        Ok(file_id)
    }
    pub fn set_workspaces(&mut self, ws: Vec<CargoWorkspace>) {
        let mut crate_graph = CrateGraph::default();
        let mut pkg_to_lib_crate = FxHashMap::default();
        let mut pkg_crates = FxHashMap::default();
        for ws in ws.iter() {
            for pkg in ws.packages() {
                for tgt in pkg.targets(ws) {
                    let root = tgt.root(ws);
                    if let Some(file_id) = self.path_map.get_id(root) {
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
        self.workspaces = Arc::new(ws);
        let mut change = AnalysisChange::new();
        change.set_crate_graph(crate_graph);
        self.analysis_host.apply_change(change);
    }
    pub fn snapshot(&self) -> ServerWorld {
        ServerWorld {
            workspaces: Arc::clone(&self.workspaces),
            analysis: self.analysis_host.analysis(),
            path_map: self.path_map.clone(),
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
        self.path_map
            .get_id(&path)
            .ok_or_else(|| format_err!("unknown file: {}", path.display()))
    }

    pub fn file_id_to_uri(&self, id: FileId) -> Result<Url> {
        let path = self.path_map.get_path(id);
        let url = Url::from_file_path(path)
            .map_err(|()| format_err!("can't convert path to url: {}", path.display()))?;
        Ok(url)
    }
}
