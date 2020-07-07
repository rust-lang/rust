//! Partitions a list of files into disjoint subsets.
//!
//! Files which do not belong to any explicitly configured `FileSet` belong to
//! the default `FileSet`.
use std::{fmt, mem};

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

// Invariant: if k1 is a prefix of k2, then they are in different buckets (k2
// is closer to 0th bucket).
// FIXME: replace with an actual trie some day.
type BadTrie<K, V> = Vec<Vec<(K, V)>>;

#[derive(Debug)]
pub struct FileSetConfig {
    n_file_sets: usize,
    trie: BadTrie<VfsPath, usize>,
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
        find_ancestor(&self.trie, path, is_prefix).copied().unwrap_or(self.len() - 1)
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

        let mut trie = BadTrie::new();

        for (i, paths) in self.roots.into_iter().enumerate() {
            for p in paths {
                insert(&mut trie, p, i, is_prefix);
            }
        }
        trie.iter_mut().for_each(|it| it.sort());
        FileSetConfig { n_file_sets, trie }
    }
}

fn is_prefix(short: &VfsPath, long: &VfsPath) -> bool {
    long.starts_with(short)
}

fn insert<K: Ord, V, P: Fn(&K, &K) -> bool>(
    trie: &mut BadTrie<K, V>,
    mut key: K,
    mut value: V,
    is_prefix: P,
) {
    'outer: for level in 0.. {
        if trie.len() == level {
            trie.push(Vec::new())
        }
        for (k, v) in trie[level].iter_mut() {
            if is_prefix(&key, k) {
                continue 'outer;
            }
            if is_prefix(k, &key) {
                mem::swap(k, &mut key);
                mem::swap(v, &mut value);
                continue 'outer;
            }
        }
        trie[level].push((key, value));
        return;
    }
}

fn find_ancestor<'t, K: Ord, V, P: Fn(&K, &K) -> bool>(
    trie: &'t BadTrie<K, V>,
    key: &K,
    is_prefix: P,
) -> Option<&'t V> {
    for bucket in trie {
        let idx = match bucket.binary_search_by(|(k, _)| k.cmp(key)) {
            Ok(it) => it,
            Err(it) => it.saturating_sub(1),
        };
        if !bucket.is_empty() && is_prefix(&bucket[idx].0, key) {
            return Some(&bucket[idx].1);
        }
    }
    None
}

#[test]
fn test_partitioning() {
    let mut file_set = FileSetConfig::builder();
    file_set.add_file_set(vec![VfsPath::new_virtual_path("/foo".into())]);
    file_set.add_file_set(vec![VfsPath::new_virtual_path("/foo/bar/baz".into())]);
    let file_set = file_set.build();

    let mut vfs = Vfs::default();
    vfs.set_file_contents(VfsPath::new_virtual_path("/foo/src/lib.rs".into()), Some(Vec::new()));
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
