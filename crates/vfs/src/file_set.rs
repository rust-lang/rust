//! Partitions a list of files into disjoint subsets.
//!
//! Files which do not belong to any explicitly configured `FileSet` belong to
//! the default `FileSet`.
use std::{cmp, fmt, iter};

use paths::AbsPathBuf;
use rustc_hash::FxHashMap;

use crate::{FileId, Vfs, VfsPath};

#[derive(Default, Clone, Eq, PartialEq)]
pub struct FileSet {
    files: FxHashMap<VfsPath, FileId>,
    paths: FxHashMap<FileId, VfsPath>,
}

impl FileSet {
    pub fn resolve_path(&self, anchor: FileId, path: &str) -> Option<FileId> {
        let mut base = self.paths[&anchor].clone();
        base.pop();
        let path = base.join(path);
        let res = self.files.get(&path).copied();
        res
    }
    pub fn insert(&mut self, file_id: FileId, path: VfsPath) {
        self.files.insert(path.clone(), file_id);
        self.paths.insert(file_id, path);
    }
    pub fn iter(&self) -> impl Iterator<Item = FileId> + '_ {
        self.paths.keys().copied()
    }
}

impl fmt::Debug for FileSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileSet").field("n_files", &self.files.len()).finish()
    }
}

#[derive(Debug)]
pub struct FileSetConfig {
    n_file_sets: usize,
    roots: Vec<(AbsPathBuf, usize)>,
}

impl FileSetConfig {
    pub fn builder() -> FileSetConfigBuilder {
        FileSetConfigBuilder::default()
    }
    pub fn partition(&self, vfs: &Vfs) -> Vec<FileSet> {
        let mut res = vec![FileSet::default(); self.len()];
        for (file_id, path) in vfs.iter() {
            let root = self.classify(&path);
            res[root].insert(file_id, path)
        }
        res
    }
    fn len(&self) -> usize {
        self.n_file_sets
    }
    fn classify(&self, path: &VfsPath) -> usize {
        for (root, idx) in self.roots.iter() {
            if let Some(path) = path.as_path() {
                if path.starts_with(root) {
                    return *idx;
                }
            }
        }
        self.len() - 1
    }
}

pub struct FileSetConfigBuilder {
    roots: Vec<Vec<AbsPathBuf>>,
}

impl Default for FileSetConfigBuilder {
    fn default() -> Self {
        FileSetConfigBuilder { roots: Vec::new() }
    }
}

impl FileSetConfigBuilder {
    pub fn add_file_set(&mut self, roots: Vec<AbsPathBuf>) {
        self.roots.push(roots)
    }
    pub fn build(self) -> FileSetConfig {
        let n_file_sets = self.roots.len() + 1;
        let mut roots: Vec<(AbsPathBuf, usize)> = self
            .roots
            .into_iter()
            .enumerate()
            .flat_map(|(i, paths)| paths.into_iter().zip(iter::repeat(i)))
            .collect();
        roots.sort_by_key(|(path, _)| cmp::Reverse(path.to_string_lossy().len()));
        FileSetConfig { n_file_sets, roots }
    }
}
