extern crate failure;
extern crate libsyntax2;
extern crate parking_lot;

use std::{
    fs,
    sync::Arc,
    collections::hash_map::HashMap,
    path::{PathBuf, Path},
};
use parking_lot::RwLock;
use libsyntax2::ast;

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;

pub struct WorldState {
    data: Arc<WorldData>
}

pub struct World {
    data: Arc<WorldData>,
}

impl WorldState {
    pub fn new() -> WorldState {
        WorldState {
            data: Arc::new(WorldData::default())
        }
    }

    pub fn snapshot(&self) -> World {
        World { data: self.data.clone() }
    }

    pub fn change_overlay(&mut self, path: PathBuf, text: Option<String>) {
        let data = self.data_mut();
        data.file_map.get_mut().remove(&path);
        data.fs_map.get_mut().remove(&path);
        if let Some(text) = text {
            data.mem_map.insert(path, Arc::new(text));
        } else {
            data.mem_map.remove(&path);
        }
    }

    fn data_mut(&mut self) -> &mut WorldData {
        if Arc::get_mut(&mut self.data).is_none() {
            let fs_map = self.data.fs_map.read().clone();
            let file_map = self.data.file_map.read().clone();
            self.data = Arc::new(WorldData {
                mem_map: self.data.mem_map.clone(),
                fs_map: RwLock::new(fs_map),
                file_map: RwLock::new(file_map),
            });
        }
        Arc::get_mut(&mut self.data).unwrap()
    }
}


impl World {
    pub fn file_syntax(&self, path: &Path) -> Result<ast::File> {
        {
            let guard = self.data.file_map.read();
            if let Some(file) = guard.get(path) {
                return Ok(file.clone());
            }
        }

        let file = self.with_file_text(path, ast::File::parse)?;
        let mut guard = self.data.file_map.write();
        let file = guard.entry(path.to_owned())
            .or_insert(file)
            .clone();
        Ok(file)
    }

    fn with_file_text<F: FnOnce(&str) -> R, R>(&self, path: &Path, f: F) -> Result<R> {
        if let Some(text) = self.data.mem_map.get(path) {
            return Ok(f(&*text));
        }

        {
            let guard = self.data.fs_map.read();
            if let Some(text) = guard.get(path) {
                return Ok(f(&*text));
            }
        }

        let text = fs::read_to_string(path)?;
        {
            let mut guard = self.data.fs_map.write();
            guard.entry(path.to_owned())
                .or_insert_with(|| Arc::new(text));
        }
        let guard = self.data.fs_map.read();
        Ok(f(&guard[path]))
    }
}


#[derive(Default)]
struct WorldData {
    mem_map: HashMap<PathBuf, Arc<String>>,
    fs_map: RwLock<HashMap<PathBuf, Arc<String>>>,
    file_map: RwLock<HashMap<PathBuf, ast::File>>,
}
