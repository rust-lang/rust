use std::{
    fs,
    path::{PathBuf, Path},
    collections::HashMap,
    sync::Arc,
};

use languageserver_types::Url;
use libanalysis::{FileId, AnalysisHost, Analysis, CrateGraph, CrateId, LibraryData};

use {
    Result,
    path_map::{PathMap, Root},
    vfs::{FileEvent, FileEventKind},
    project_model::CargoWorkspace,
};

#[derive(Debug)]
pub struct ServerWorldState {
    pub workspaces: Arc<Vec<CargoWorkspace>>,
    pub analysis_host: AnalysisHost,
    pub path_map: PathMap,
    pub mem_map: HashMap<FileId, Option<String>>,
}

#[derive(Clone)]
pub struct ServerWorld {
    pub workspaces: Arc<Vec<CargoWorkspace>>,
    pub analysis: Analysis,
    pub path_map: PathMap,
}

impl ServerWorldState {
    pub fn new() -> ServerWorldState {
        ServerWorldState {
            workspaces: Arc::new(Vec::new()),
            analysis_host: AnalysisHost::new(),
            path_map: PathMap::new(),
            mem_map: HashMap::new(),
        }
    }

    pub fn apply_fs_changes(&mut self, events: Vec<FileEvent>) {
        let pm = &mut self.path_map;
        let mm = &mut self.mem_map;
        let changes = events.into_iter()
            .map(|event| {
                let text = match event.kind {
                    FileEventKind::Add(text) => Some(text),
                };
                (event.path, text)
            })
            .map(|(path, text)| {
                (pm.get_or_insert(path, Root::Workspace), text)
            })
            .filter_map(|(id, text)| {
                if mm.contains_key(&id) {
                    mm.insert(id, text);
                    None
                } else {
                    Some((id, text))
                }
            });

        self.analysis_host.change_files(changes);
    }
    pub fn events_to_files(&mut self, events: Vec<FileEvent>) -> Vec<(FileId, String)> {
        let pm = &mut self.path_map;
        events.into_iter()
            .map(|event| {
                let text = match event.kind {
                    FileEventKind::Add(text) => text,
                };
                (event.path, text)
            })
            .map(|(path, text)| (pm.get_or_insert(path, Root::Lib), text))
            .collect()
    }
    pub fn add_lib(&mut self, data: LibraryData) {
        self.analysis_host.add_library(data);
    }

    pub fn add_mem_file(&mut self, path: PathBuf, text: String) -> FileId {
        let file_id = self.path_map.get_or_insert(path, Root::Workspace);
        self.mem_map.insert(file_id, None);
        if self.path_map.get_root(file_id) != Root::Lib {
            self.analysis_host.change_file(file_id, Some(text));
        }
        file_id
    }

    pub fn change_mem_file(&mut self, path: &Path, text: String) -> Result<()> {
        let file_id = self.path_map.get_id(path).ok_or_else(|| {
            format_err!("change to unknown file: {}", path.display())
        })?;
        if self.path_map.get_root(file_id) != Root::Lib {
            self.analysis_host.change_file(file_id, Some(text));
        }
        Ok(())
    }

    pub fn remove_mem_file(&mut self, path: &Path) -> Result<FileId> {
        let file_id = self.path_map.get_id(path).ok_or_else(|| {
            format_err!("change to unknown file: {}", path.display())
        })?;
        match self.mem_map.remove(&file_id) {
            Some(_) => (),
            None => bail!("unmatched close notification"),
        };
        // Do this via file watcher ideally.
        let text = fs::read_to_string(path).ok();
        if self.path_map.get_root(file_id) != Root::Lib {
            self.analysis_host.change_file(file_id, text);
        }
        Ok(file_id)
    }
    pub fn set_workspaces(&mut self, ws: Vec<CargoWorkspace>) {
        let mut crate_roots = HashMap::new();
        ws.iter()
          .flat_map(|ws| {
              ws.packages()
                .flat_map(move |pkg| pkg.targets(ws))
                .map(move |tgt| tgt.root(ws))
          })
          .for_each(|root| {
              if let Some(file_id) = self.path_map.get_id(root) {
                  let crate_id = CrateId(crate_roots.len() as u32);
                  crate_roots.insert(crate_id, file_id);
              }
          });
        let crate_graph = CrateGraph { crate_roots };
        self.workspaces = Arc::new(ws);
        self.analysis_host.set_crate_graph(crate_graph);
    }
    pub fn snapshot(&self) -> ServerWorld {
        ServerWorld {
            workspaces: Arc::clone(&self.workspaces),
            analysis: self.analysis_host.analysis(self.path_map.clone()),
            path_map: self.path_map.clone()
        }
    }
}

impl ServerWorld {
    pub fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    pub fn uri_to_file_id(&self, uri: &Url) -> Result<FileId> {
        let path = uri.to_file_path()
            .map_err(|()| format_err!("invalid uri: {}", uri))?;
        self.path_map.get_id(&path).ok_or_else(|| format_err!("unknown file: {}", path.display()))
    }

    pub fn file_id_to_uri(&self, id: FileId) -> Result<Url> {
        let path = self.path_map.get_path(id);
        let url = Url::from_file_path(path)
            .map_err(|()| format_err!("can't convert path to url: {}", path.display()))?;
        Ok(url)
    }
}
