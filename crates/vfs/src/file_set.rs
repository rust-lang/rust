//! Partitions a list of files into disjoint subsets.
//!
//! Files which do not belong to any explicitly configured `FileSet` belong to
//! the default `FileSet`.
use std::fmt;

use fst::{IntoStreamer, Streamer};
use rustc_hash::FxHashMap;

use crate::{FileId, Vfs, VfsPath};

#[derive(Default, Clone, Eq, PartialEq)]
pub struct FileSet {
    files: FxHashMap<VfsPath, FileId>,
    paths: FxHashMap<FileId, VfsPath>,
}

impl FileSet {
    pub fn len(&self) -> usize {
        self.files.len()
    }
    pub fn resolve_path(&self, anchor: FileId, path: &str) -> Option<FileId> {
        let mut base = self.paths[&anchor].clone();
        base.pop();
        let path = base.join(path)?;
        self.files.get(&path).copied()
    }

    pub fn file_for_path(&self, path: &VfsPath) -> Option<&FileId> {
        self.files.get(path)
    }

    pub fn path_for_file(&self, file: &FileId) -> Option<&VfsPath> {
        self.paths.get(file)
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
    map: fst::Map<Vec<u8>>,
}

impl Default for FileSetConfig {
    fn default() -> Self {
        FileSetConfig::builder().build()
    }
}

impl FileSetConfig {
    pub fn builder() -> FileSetConfigBuilder {
        FileSetConfigBuilder::default()
    }
    pub fn partition(&self, vfs: &Vfs) -> Vec<FileSet> {
        let mut scratch_space = Vec::new();
        let mut res = vec![FileSet::default(); self.len()];
        for (file_id, path) in vfs.iter() {
            let root = self.classify(&path, &mut scratch_space);
            res[root].insert(file_id, path.clone())
        }
        res
    }
    fn len(&self) -> usize {
        self.n_file_sets
    }
    fn classify(&self, path: &VfsPath, scratch_space: &mut Vec<u8>) -> usize {
        scratch_space.clear();
        path.encode(scratch_space);
        let automaton = PrefixOf::new(scratch_space.as_slice());
        let mut longest_prefix = self.len() - 1;
        let mut stream = self.map.search(automaton).into_stream();
        while let Some((_, v)) = stream.next() {
            longest_prefix = v as usize;
        }
        longest_prefix
    }
}

pub struct FileSetConfigBuilder {
    roots: Vec<Vec<VfsPath>>,
}

impl Default for FileSetConfigBuilder {
    fn default() -> Self {
        FileSetConfigBuilder { roots: Vec::new() }
    }
}

impl FileSetConfigBuilder {
    pub fn len(&self) -> usize {
        self.roots.len()
    }
    pub fn add_file_set(&mut self, roots: Vec<VfsPath>) {
        self.roots.push(roots)
    }
    pub fn build(self) -> FileSetConfig {
        let n_file_sets = self.roots.len() + 1;
        let map = {
            let mut entries = Vec::new();
            for (i, paths) in self.roots.into_iter().enumerate() {
                for p in paths {
                    let mut buf = Vec::new();
                    p.encode(&mut buf);
                    entries.push((buf, i as u64));
                }
            }
            entries.sort();
            entries.dedup_by(|(a, _), (b, _)| a == b);
            fst::Map::from_iter(entries).unwrap()
        };
        FileSetConfig { n_file_sets, map }
    }
}

struct PrefixOf<'a> {
    prefix_of: &'a [u8],
}

impl<'a> PrefixOf<'a> {
    fn new(prefix_of: &'a [u8]) -> Self {
        Self { prefix_of }
    }
}

impl fst::Automaton for PrefixOf<'_> {
    type State = usize;
    fn start(&self) -> usize {
        0
    }
    fn is_match(&self, &state: &usize) -> bool {
        state != !0
    }
    fn can_match(&self, &state: &usize) -> bool {
        state != !0
    }
    fn accept(&self, &state: &usize, byte: u8) -> usize {
        if self.prefix_of.get(state) == Some(&byte) {
            state + 1
        } else {
            !0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_prefix() {
        let mut file_set = FileSetConfig::builder();
        file_set.add_file_set(vec![VfsPath::new_virtual_path("/foo".into())]);
        file_set.add_file_set(vec![VfsPath::new_virtual_path("/foo/bar/baz".into())]);
        let file_set = file_set.build();

        let mut vfs = Vfs::default();
        vfs.set_file_contents(
            VfsPath::new_virtual_path("/foo/src/lib.rs".into()),
            Some(Vec::new()),
        );
        vfs.set_file_contents(
            VfsPath::new_virtual_path("/foo/src/bar/baz/lib.rs".into()),
            Some(Vec::new()),
        );
        vfs.set_file_contents(
            VfsPath::new_virtual_path("/foo/bar/baz/lib.rs".into()),
            Some(Vec::new()),
        );
        vfs.set_file_contents(VfsPath::new_virtual_path("/quux/lib.rs".into()), Some(Vec::new()));

        let partition = file_set.partition(&vfs).into_iter().map(|it| it.len()).collect::<Vec<_>>();
        assert_eq!(partition, vec![2, 1, 1]);
    }

    #[test]
    fn name_prefix() {
        let mut file_set = FileSetConfig::builder();
        file_set.add_file_set(vec![VfsPath::new_virtual_path("/foo".into())]);
        file_set.add_file_set(vec![VfsPath::new_virtual_path("/foo-things".into())]);
        let file_set = file_set.build();

        let mut vfs = Vfs::default();
        vfs.set_file_contents(
            VfsPath::new_virtual_path("/foo/src/lib.rs".into()),
            Some(Vec::new()),
        );
        vfs.set_file_contents(
            VfsPath::new_virtual_path("/foo-things/src/lib.rs".into()),
            Some(Vec::new()),
        );

        let partition = file_set.partition(&vfs).into_iter().map(|it| it.len()).collect::<Vec<_>>();
        assert_eq!(partition, vec![1, 1, 0]);
    }
}
