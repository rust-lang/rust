//! Thin wrappers around `std::path`, distinguishing between absolute and
//! relative paths.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

use std::{
    borrow::Borrow,
    ffi::OsStr,
    fmt, ops,
    path::{Component, Path, PathBuf},
};

/// Wrapper around an absolute [`PathBuf`].
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

impl Borrow<AbsPath> for AbsPathBuf {
    fn borrow(&self) -> &AbsPath {
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
    /// Wrap the given absolute path in `AbsPathBuf`
    ///
    /// # Panics
    ///
    /// Panics if `path` is not absolute.
    pub fn assert(path: PathBuf) -> AbsPathBuf {
        AbsPathBuf::try_from(path)
            .unwrap_or_else(|path| panic!("expected absolute path, got {}", path.display()))
    }

    /// Coerces to an `AbsPath` slice.
    ///
    /// Equivalent of [`PathBuf::as_path`] for `AbsPathBuf`.
    pub fn as_path(&self) -> &AbsPath {
        AbsPath::assert(self.0.as_path())
    }

    /// Equivalent of [`PathBuf::pop`] for `AbsPathBuf`.
    ///
    /// Note that this won't remove the root component, so `self` will still be
    /// absolute.
    pub fn pop(&mut self) -> bool {
        self.0.pop()
    }
}

impl fmt::Display for AbsPathBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0.display(), f)
    }
}

/// Wrapper around an absolute [`Path`].
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct AbsPath(Path);

impl AsRef<Path> for AbsPath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl ToOwned for AbsPath {
    type Owned = AbsPathBuf;

    fn to_owned(&self) -> Self::Owned {
        AbsPathBuf(self.0.to_owned())
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
    /// Wrap the given absolute path in `AbsPath`
    ///
    /// # Panics
    ///
    /// Panics if `path` is not absolute.
    pub fn assert(path: &Path) -> &AbsPath {
        assert!(path.is_absolute());
        unsafe { &*(path as *const Path as *const AbsPath) }
    }

    /// Equivalent of [`Path::parent`] for `AbsPath`.
    pub fn parent(&self) -> Option<&AbsPath> {
        self.0.parent().map(AbsPath::assert)
    }

    /// Equivalent of [`Path::join`] for `AbsPath` with an additional normalize step afterwards.
    pub fn absolutize(&self, path: impl AsRef<Path>) -> AbsPathBuf {
        self.join(path).normalize()
    }

    /// Equivalent of [`Path::join`] for `AbsPath`.
    pub fn join(&self, path: impl AsRef<Path>) -> AbsPathBuf {
        self.as_ref().join(path).try_into().unwrap()
    }

    /// Normalize the given path:
    /// - Removes repeated separators: `/a//b` becomes `/a/b`
    /// - Removes occurrences of `.` and resolves `..`.
    /// - Removes trailing slashes: `/a/b/` becomes `/a/b`.
    ///
    /// # Example
    /// ```
    /// # use paths::AbsPathBuf;
    /// let abs_path_buf = AbsPathBuf::assert("/a/../../b/.//c//".into());
    /// let normalized = abs_path_buf.normalize();
    /// assert_eq!(normalized, AbsPathBuf::assert("/b/c".into()));
    /// ```
    pub fn normalize(&self) -> AbsPathBuf {
        AbsPathBuf(normalize_path(&self.0))
    }

    /// Equivalent of [`Path::to_path_buf`] for `AbsPath`.
    pub fn to_path_buf(&self) -> AbsPathBuf {
        AbsPathBuf::try_from(self.0.to_path_buf()).unwrap()
    }

    pub fn canonicalize(&self) -> ! {
        panic!("We explicitly do not provide canonicalization API, as that is almost always a wrong solution, see #14430")
    }

    /// Equivalent of [`Path::strip_prefix`] for `AbsPath`.
    ///
    /// Returns a relative path.
    pub fn strip_prefix(&self, base: &AbsPath) -> Option<&RelPath> {
        self.0.strip_prefix(base).ok().map(RelPath::new_unchecked)
    }
    pub fn starts_with(&self, base: &AbsPath) -> bool {
        self.0.starts_with(&base.0)
    }
    pub fn ends_with(&self, suffix: &RelPath) -> bool {
        self.0.ends_with(&suffix.0)
    }

    pub fn name_and_extension(&self) -> Option<(&str, Option<&str>)> {
        Some((
            self.file_stem()?.to_str()?,
            self.extension().and_then(|extension| extension.to_str()),
        ))
    }

    // region:delegate-methods

    // Note that we deliberately don't implement `Deref<Target = Path>` here.
    //
    // The problem with `Path` is that it directly exposes convenience IO-ing
    // methods. For example, `Path::exists` delegates to `fs::metadata`.
    //
    // For `AbsPath`, we want to make sure that this is a POD type, and that all
    // IO goes via `fs`. That way, it becomes easier to mock IO when we need it.

    pub fn file_name(&self) -> Option<&OsStr> {
        self.0.file_name()
    }
    pub fn extension(&self) -> Option<&OsStr> {
        self.0.extension()
    }
    pub fn file_stem(&self) -> Option<&OsStr> {
        self.0.file_stem()
    }
    pub fn as_os_str(&self) -> &OsStr {
        self.0.as_os_str()
    }
    #[deprecated(note = "use Display instead")]
    pub fn display(&self) -> std::path::Display<'_> {
        self.0.display()
    }
    #[deprecated(note = "use std::fs::metadata().is_ok() instead")]
    pub fn exists(&self) -> bool {
        self.0.exists()
    }
    // endregion:delegate-methods
}

impl fmt::Display for AbsPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0.display(), f)
    }
}

/// Wrapper around a relative [`PathBuf`].
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
    /// Coerces to a `RelPath` slice.
    ///
    /// Equivalent of [`PathBuf::as_path`] for `RelPathBuf`.
    pub fn as_path(&self) -> &RelPath {
        RelPath::new_unchecked(self.0.as_path())
    }
}

/// Wrapper around a relative [`Path`].
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct RelPath(Path);

impl AsRef<Path> for RelPath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl RelPath {
    /// Creates a new `RelPath` from `path`, without checking if it is relative.
    pub fn new_unchecked(path: &Path) -> &RelPath {
        unsafe { &*(path as *const Path as *const RelPath) }
    }
}

/// Taken from <https://github.com/rust-lang/cargo/blob/79c769c3d7b4c2cf6a93781575b7f592ef974255/src/cargo/util/paths.rs#L60-L85>
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = path.components().peekable();
    let mut ret = if let Some(c @ Component::Prefix(..)) = components.peek().copied() {
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
