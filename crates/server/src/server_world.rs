use std::{
    path::{PathBuf, Path},
    collections::HashMap,
};

use languageserver_types::Url;
use libanalysis::{FileId, WorldState, World};

use {
    Result,
    path_map::PathMap,
    vfs::{FileEvent, FileEventKind},
};

#[derive(Debug)]
pub struct ServerWorldState {
    pub analysis: WorldState,
    pub path_map: PathMap,
    pub mem_map: HashMap<FileId, Option<String>>,
}

#[derive(Clone)]
pub struct ServerWorld {
    pub analysis: World,
    pub path_map: PathMap,
}

impl ServerWorldState {
    pub fn new() -> ServerWorldState {
        ServerWorldState {
            analysis: WorldState::new(),
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
                    FileEventKind::Remove => None,
                };
                (event.path, text)
            })
            .map(|(path, text)| {
                (pm.get_or_insert(path), text)
            })
            .filter_map(|(id, text)| {
                if mm.contains_key(&id) {
                    mm.insert(id, text);
                    None
                } else {
                    Some((id, text))
                }
            });

        self.analysis.change_files(changes);
    }

    pub fn add_mem_file(&mut self, path: PathBuf, text: String) {
        let file_id = self.path_map.get_or_insert(path);
        self.mem_map.insert(file_id, None);
        self.analysis.change_file(file_id, Some(text));
    }

    pub fn change_mem_file(&mut self, path: &Path, text: String) -> Result<()> {
        let file_id = self.path_map.get_id(path).ok_or_else(|| {
            format_err!("change to unknown file: {}", path.display())
        })?;
        self.analysis.change_file(file_id, Some(text));
        Ok(())
    }

    pub fn remove_mem_file(&mut self, path: &Path) -> Result<()> {
        let file_id = self.path_map.get_id(path).ok_or_else(|| {
            format_err!("change to unknown file: {}", path.display())
        })?;
        let text = match self.mem_map.remove(&file_id) {
            Some(text) => text,
            None => bail!("unmatched close notification"),
        };
        self.analysis.change_file(file_id, text);
        Ok(())
    }

    pub fn snapshot(&self) -> ServerWorld {
        ServerWorld {
            analysis: self.analysis.snapshot(self.path_map.clone()),
            path_map: self.path_map.clone()
        }
    }
}

impl ServerWorld {
    pub fn analysis(&self) -> &World {
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
