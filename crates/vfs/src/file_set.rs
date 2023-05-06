//! Partitions a list of files into disjoint subsets.
//!
//! Files which do not belong to any explicitly configured `FileSet` belong to
//! the default `FileSet`.
use std::fmt;

use fst::{IntoStreamer, Streamer};
use nohash_hasher::IntMap;
use rustc_hash::FxHashMap;

use crate::{AnchoredPath, FileId, Vfs, VfsPath};

/// A set of [`VfsPath`]s identified by [`FileId`]s.
#[derive(Default, Clone, Eq, PartialEq)]
pub struct FileSet {
    files: FxHashMap<VfsPath, FileId>,
    paths: IntMap<FileId, VfsPath>,
}

impl FileSet {
    /// Returns the number of stored paths.
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Get the id of the file corresponding to `path`.
    ///
    /// If either `path`'s [`anchor`](AnchoredPath::anchor) or the resolved path is not in
    /// the set, returns [`None`].
    pub fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId> {
        let mut base = self.paths[&path.anchor].clone();
        base.pop();
        let path = base.join(path.path)?;
        self.files.get(&path).copied()
    }

    /// Get the id corresponding to `path` if it exists in the set.
    pub fn file_for_path(&self, path: &VfsPath) -> Option<&FileId> {
        self.files.get(path)
    }

    /// Get the path corresponding to `file` if it exists in the set.
    pub fn path_for_file(&self, file: &FileId) -> Option<&VfsPath> {
        self.paths.get(file)
    }

    /// Insert the `file_id, path` pair into the set.
    ///
    /// # Note
    /// Multiple [`FileId`] can be mapped to the same [`VfsPath`], and vice-versa.
    pub fn insert(&mut self, file_id: FileId, path: VfsPath) {
        self.files.insert(path.clone(), file_id);
        self.paths.insert(file_id, path);
    }

    /// Iterate over this set's ids.
    pub fn iter(&self) -> impl Iterator<Item = FileId> + '_ {
        self.paths.keys().copied()
    }
}

impl fmt::Debug for FileSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileSet").field("n_files", &self.files.len()).finish()
    }
}

/// This contains path prefixes to partition a [`Vfs`] into [`FileSet`]s.
///
/// # Example
/// ```rust
/// # use vfs::{file_set::FileSetConfigBuilder, VfsPath, Vfs};
/// let mut builder = FileSetConfigBuilder::default();
/// builder.add_file_set(vec![VfsPath::new_virtual_path("/src".to_string())]);
/// let config = builder.build();
/// let mut file_system = Vfs::default();
/// file_system.set_file_contents(VfsPath::new_virtual_path("/src/main.rs".to_string()), Some(vec![]));
/// file_system.set_file_contents(VfsPath::new_virtual_path("/src/lib.rs".to_string()), Some(vec![]));
/// file_system.set_file_contents(VfsPath::new_virtual_path("/build.rs".to_string()), Some(vec![]));
/// // contains the sets :
/// // { "/src/main.rs", "/src/lib.rs" }
/// // { "build.rs" }
/// let sets = config.partition(&file_system);
/// ```
#[derive(Debug)]
pub struct FileSetConfig {
    /// Number of sets that `self` can partition a [`Vfs`] into.
    ///
    /// This should be the number of sets in `self.map` + 1 for files that don't fit in any
    /// defined set.
    n_file_sets: usize,
    /// Map from encoded paths to the set they belong to.
    map: fst::Map<Vec<u8>>,
}

impl Default for FileSetConfig {
    fn default() -> Self {
        FileSetConfig::builder().build()
    }
}

impl FileSetConfig {
    /// Returns a builder for `FileSetConfig`.
    pub fn builder() -> FileSetConfigBuilder {
        FileSetConfigBuilder::default()
    }

    /// Partition `vfs` into `FileSet`s.
    ///
    /// Creates a new [`FileSet`] for every set of prefixes in `self`.
    pub fn partition(&self, vfs: &Vfs) -> Vec<FileSet> {
        let mut scratch_space = Vec::new();
        let mut res = vec![FileSet::default(); self.len()];
        for (file_id, path) in vfs.iter() {
            let root = self.classify(path, &mut scratch_space);
            res[root].insert(file_id, path.clone());
        }
        res
    }

    /// Number of sets that `self` can partition a [`Vfs`] into.
    fn len(&self) -> usize {
        self.n_file_sets
    }

    /// Returns the set index for the given `path`.
    ///
    /// `scratch_space` is used as a buffer and will be entirely replaced.
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

/// Builder for [`FileSetConfig`].
#[derive(Default)]
pub struct FileSetConfigBuilder {
    roots: Vec<Vec<VfsPath>>,
}

impl FileSetConfigBuilder {
    /// Returns the number of sets currently held.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Add a new set of paths prefixes.
    pub fn add_file_set(&mut self, roots: Vec<VfsPath>) {
        self.roots.push(roots);
    }

    /// Build the `FileSetConfig`.
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

/// Implements [`fst::Automaton`]
///
/// It will match if `prefix_of` is a prefix of the given data.
struct PrefixOf<'a> {
    prefix_of: &'a [u8],
}

impl<'a> PrefixOf<'a> {
    /// Creates a new `PrefixOf` from the given slice.
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
mod tests;
