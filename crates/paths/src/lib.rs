//! Thin wrappers around `std::path`, distinguishing between absolute and
//! relative paths.
use std::{
    convert::{TryFrom, TryInto},
    ops,
    path::{Component, Path, PathBuf},
};

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AbsPathBuf(PathBuf);

impl From<AbsPathBuf> for PathBuf {
    fn from(AbsPathBuf(path_buf): AbsPathBuf) -> PathBuf {
        path_buf
    }
}

impl ops::Deref for AbsPathBuf {
    type Target = AbsPath;
    fn deref(&self) -> &AbsPath {
        self.as_path()
    }
}

impl AsRef<Path> for AbsPathBuf {
    fn as_ref(&self) -> &Path {
        self.0.as_path()
    }
}

impl AsRef<AbsPath> for AbsPathBuf {
    fn as_ref(&self) -> &AbsPath {
        self.as_path()
    }
}

impl TryFrom<PathBuf> for AbsPathBuf {
    type Error = PathBuf;
    fn try_from(path_buf: PathBuf) -> Result<AbsPathBuf, PathBuf> {
        if !path_buf.is_absolute() {
            return Err(path_buf);
        }
        Ok(AbsPathBuf(path_buf))
    }
}

impl TryFrom<&str> for AbsPathBuf {
    type Error = PathBuf;
    fn try_from(path: &str) -> Result<AbsPathBuf, PathBuf> {
        AbsPathBuf::try_from(PathBuf::from(path))
    }
}

impl PartialEq<AbsPath> for AbsPathBuf {
    fn eq(&self, other: &AbsPath) -> bool {
        self.as_path() == other
    }
}

impl AbsPathBuf {
    pub fn assert(path: PathBuf) -> AbsPathBuf {
        AbsPathBuf::try_from(path)
            .unwrap_or_else(|path| panic!("expected absolute path, got {}", path.display()))
    }
    pub fn as_path(&self) -> &AbsPath {
        AbsPath::assert(self.0.as_path())
    }
    pub fn pop(&mut self) -> bool {
        self.0.pop()
    }
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct AbsPath(Path);

impl ops::Deref for AbsPath {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for AbsPath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl<'a> TryFrom<&'a Path> for &'a AbsPath {
    type Error = &'a Path;
    fn try_from(path: &'a Path) -> Result<&'a AbsPath, &'a Path> {
        if !path.is_absolute() {
            return Err(path);
        }
        Ok(AbsPath::assert(path))
    }
}

impl AbsPath {
    pub fn assert(path: &Path) -> &AbsPath {
        assert!(path.is_absolute());
        unsafe { &*(path as *const Path as *const AbsPath) }
    }

    pub fn parent(&self) -> Option<&AbsPath> {
        self.0.parent().map(AbsPath::assert)
    }
    pub fn join(&self, path: impl AsRef<Path>) -> AbsPathBuf {
        self.as_ref().join(path).try_into().unwrap()
    }
    pub fn normalize(&self) -> AbsPathBuf {
        AbsPathBuf(normalize_path(&self.0))
    }
    pub fn to_path_buf(&self) -> AbsPathBuf {
        AbsPathBuf::try_from(self.0.to_path_buf()).unwrap()
    }
    pub fn strip_prefix(&self, base: &AbsPath) -> Option<&RelPath> {
        self.0.strip_prefix(base).ok().map(RelPath::new_unchecked)
    }
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct RelPathBuf(PathBuf);

impl From<RelPathBuf> for PathBuf {
    fn from(RelPathBuf(path_buf): RelPathBuf) -> PathBuf {
        path_buf
    }
}

impl ops::Deref for RelPathBuf {
    type Target = RelPath;
    fn deref(&self) -> &RelPath {
        self.as_path()
    }
}

impl AsRef<Path> for RelPathBuf {
    fn as_ref(&self) -> &Path {
        self.0.as_path()
    }
}

impl TryFrom<PathBuf> for RelPathBuf {
    type Error = PathBuf;
    fn try_from(path_buf: PathBuf) -> Result<RelPathBuf, PathBuf> {
        if !path_buf.is_relative() {
            return Err(path_buf);
        }
        Ok(RelPathBuf(path_buf))
    }
}

impl TryFrom<&str> for RelPathBuf {
    type Error = PathBuf;
    fn try_from(path: &str) -> Result<RelPathBuf, PathBuf> {
        RelPathBuf::try_from(PathBuf::from(path))
    }
}

impl RelPathBuf {
    pub fn as_path(&self) -> &RelPath {
        RelPath::new_unchecked(self.0.as_path())
    }
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct RelPath(Path);

impl ops::Deref for RelPath {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for RelPath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl RelPath {
    pub fn new_unchecked(path: &Path) -> &RelPath {
        unsafe { &*(path as *const Path as *const RelPath) }
    }
}

// https://github.com/rust-lang/cargo/blob/79c769c3d7b4c2cf6a93781575b7f592ef974255/src/cargo/util/paths.rs#L60-L85
fn normalize_path(path: &Path) -> PathBuf {
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
