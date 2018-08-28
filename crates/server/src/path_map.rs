use std::path::{PathBuf, Path, Component};
use im;
use relative_path::RelativePath;
use libanalysis::{FileId, FileResolver};

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

impl FileResolver for PathMap {
    fn file_stem(&self, id: FileId) -> String {
        self.get_path(id).file_stem().unwrap().to_str().unwrap().to_string()
    }

    fn resolve(&self, id: FileId, path: &RelativePath) -> Option<FileId> {
        let path = path.to_path(&self.get_path(id));
        let path = normalize(&path);
        self.get_id(&path)
    }
}

fn normalize(path: &Path) -> PathBuf {
    let mut components = path.components().peekable();
    let mut ret = if let Some(c @ Component::Prefix(..)) = components.peek().cloned() {
        components.next();
        PathBuf::from(c.as_os_str())
    } else {
        PathBuf::new()
    };

    for component in components {
        match component {
            Component::Prefix(..) => unreachable!(),
            Component::RootDir => {
                ret.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                ret.pop();
            }
            Component::Normal(c) => {
                ret.push(c);
            }
        }
    }
    ret
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_resolve() {
        let mut m = PathMap::new();
        let id1 = m.get_or_insert(PathBuf::from("/foo"));
        let id2 = m.get_or_insert(PathBuf::from("/foo/bar.rs"));
        assert_eq!(
            m.resolve(id1, &RelativePath::new("bar.rs")),
            Some(id2),
        )
    }
}

