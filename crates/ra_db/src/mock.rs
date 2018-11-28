use std::sync::Arc;

use rustc_hash::FxHashSet;
use relative_path::{RelativePath, RelativePathBuf};

use crate::{FileId, FileResolver, SourceRoot, FileResolverImp};

#[derive(Default, Debug)]
pub struct FileMap(Vec<(FileId, RelativePathBuf)>);

impl FileMap {
    pub fn add(&mut self, path: RelativePathBuf) -> FileId {
        let file_id = FileId((self.0.len() + 1) as u32);
        self.0.push((file_id, path));
        file_id
    }

    pub fn into_source_root(self) -> SourceRoot {
        let files = self.files();
        let file_resolver = FileResolverImp::new(Arc::new(self));
        SourceRoot {
            file_resolver,
            files,
        }
    }

    pub fn files(&self) -> FxHashSet<FileId> {
        self.iter().map(|(id, _)| id).collect()
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (FileId, &'a RelativePath)> + 'a {
        self.0
            .iter()
            .map(|(id, path)| (*id, path.as_relative_path()))
    }

    fn path(&self, id: FileId) -> &RelativePath {
        self.iter().find(|&(it, _)| it == id).unwrap().1
    }
}

impl FileResolver for FileMap {
    fn file_stem(&self, id: FileId) -> String {
        self.path(id).file_stem().unwrap().to_string()
    }
    fn resolve(&self, id: FileId, rel: &RelativePath) -> Option<FileId> {
        let path = self.path(id).join(rel).normalize();
        let id = self.iter().find(|&(_, p)| path == p)?.0;
        Some(id)
    }
}
