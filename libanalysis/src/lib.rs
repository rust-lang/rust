extern crate failure;
extern crate parking_lot;
#[macro_use]
extern crate log;
extern crate once_cell;
extern crate libsyntax2;
extern crate libeditor;

use once_cell::sync::OnceCell;

use std::{
    fs,
    sync::Arc,
    collections::hash_map::HashMap,
    path::{PathBuf, Path},
};
use parking_lot::RwLock;
use libsyntax2::ast;
use libeditor::LineIndex;

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
        if let Some(text) = text {
            data.mem_map.insert(path, Arc::new(text));
        } else {
            data.mem_map.remove(&path);
        }
    }

    fn data_mut(&mut self) -> &mut WorldData {
        if Arc::get_mut(&mut self.data).is_none() {
            let file_map = self.data.file_map.read().clone();
            self.data = Arc::new(WorldData {
                mem_map: self.data.mem_map.clone(),
                file_map: RwLock::new(file_map),
            });
        }
        Arc::get_mut(&mut self.data).unwrap()
    }
}


impl World {
    pub fn file_syntax(&self, path: &Path) -> Result<ast::File> {
        let data = self.file_data(path)?;
        let syntax = data.syntax
            .get_or_init(|| {
                trace!("parsing: {}", path.display());
                ast::File::parse(self.file_text(path, &data))
            }).clone();
        Ok(syntax)
    }

    pub fn file_line_index(&self, path: &Path) -> Result<LineIndex> {
        let data = self.file_data(path)?;
        let index = data.lines
            .get_or_init(|| {
                trace!("calc line index: {}", path.display());
                LineIndex::new(self.file_text(path, &data))
            });
        Ok(index.clone())
    }

    fn file_text<'a>(&'a self, path: &Path, file_data: &'a FileData) -> &'a str {
        match file_data.text.as_ref() {
            Some(text) => text.as_str(),
            None => self.data.mem_map[path].as_str()
        }
    }

    fn file_data(&self, path: &Path) -> Result<Arc<FileData>> {
        {
            let guard = self.data.file_map.read();
            if let Some(data) = guard.get(path) {
                return Ok(data.clone());
            }
        }

        let text = if self.data.mem_map.contains_key(path) {
            None
        } else {
            trace!("loading file from disk: {}", path.display());
            Some(fs::read_to_string(path)?)
        };
        let res = {
            let mut guard = self.data.file_map.write();
            guard.entry(path.to_owned())
                .or_insert_with(|| Arc::new(FileData {
                    text,
                    syntax: OnceCell::new(),
                    lines: OnceCell::new(),
                }))
                .clone()
        };
        Ok(res)
    }
}


#[derive(Default)]
struct WorldData {
    mem_map: HashMap<PathBuf, Arc<String>>,
    file_map: RwLock<HashMap<PathBuf, Arc<FileData>>>,
}

struct FileData {
    text: Option<String>,
    syntax: OnceCell<ast::File>,
    lines: OnceCell<LineIndex>,
}
