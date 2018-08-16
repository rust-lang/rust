use std::path::{PathBuf, Path};
use im;
use libanalysis::{FileId};

#[derive(Debug, Default, Clone)]
pub struct PathMap {
    next_id: u32,
    path2id: im::HashMap<PathBuf, FileId>,
    id2path: im::HashMap<FileId, PathBuf>,
}

impl PathMap {
    pub fn new() -> PathMap {
        Default::default()
    }

    pub fn get_or_insert(&mut self, path: PathBuf) -> FileId {
        self.path2id.get(path.as_path())
            .map(|&id| id)
            .unwrap_or_else(|| {
                let id = self.new_file_id();
                self.insert(path, id);
                id
            })
    }

    pub fn get_id(&self, path: &Path) -> Option<FileId> {
        self.path2id.get(path).map(|&id| id)
    }

    pub fn get_path(&self, id: FileId) -> &Path {
        self.id2path.get(&id)
            .unwrap()
            .as_path()
    }

    pub fn resolve(&self, id: FileId, relpath: &Path) -> Option<FileId> {
        let path = self.get_path(id).join(relpath);
        self.get_id(&path)
    }

    fn insert(&mut self, path: PathBuf, id: FileId) {
        self.path2id.insert(path.clone(), id);
        self.id2path.insert(id, path.clone());
    }

    fn new_file_id(&mut self) -> FileId {
        let id = FileId(self.next_id);
        self.next_id += 1;
        id
    }
}
