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

impl AbsPathBuf {
    pub fn as_path(&self) -> &AbsPath {
        AbsPath::new_unchecked(self.0.as_path())
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
        Ok(AbsPath::new_unchecked(path))
    }
}

impl AbsPath {
    fn new_unchecked(path: &Path) -> &AbsPath {
        unsafe { &*(path as *const Path as *const AbsPath) }
    }

    pub fn join(&self, path: impl AsRef<Path>) -> AbsPathBuf {
        self.as_ref().join(path).try_into().unwrap()
    }
    pub fn normalize(&self) -> AbsPathBuf {
        AbsPathBuf(normalize_path(&self.0))
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
