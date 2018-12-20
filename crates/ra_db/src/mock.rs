use rustc_hash::FxHashSet;
use relative_path::{RelativePath, RelativePathBuf};

use crate::{FileId};

#[derive(Default, Debug, Clone)]
pub struct FileMap(Vec<(FileId, RelativePathBuf)>);

impl FileMap {
    pub fn add(&mut self, path: RelativePathBuf) -> FileId {
        let file_id = FileId((self.0.len() + 1) as u32);
        self.0.push((file_id, path));
        file_id
    }

    pub fn files(&self) -> FxHashSet<FileId> {
        self.iter().map(|(id, _)| id).collect()
    }

    pub fn file_id(&self, path: &str) -> FileId {
        assert!(path.starts_with('/'));
        self.iter().find(|(_, p)| p == &path[1..]).unwrap().0
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (FileId, &'a RelativePath)> + 'a {
        self.0
            .iter()
            .map(|(id, path)| (*id, path.as_relative_path()))
    }
}
