use std::{
    sync::Arc,
    hash::{Hash, Hasher},
    fmt,
};

use relative_path::RelativePath;

use crate::input::FileId;

pub trait FileResolver: fmt::Debug + Send + Sync + 'static {
    fn file_stem(&self, file_id: FileId) -> String;
    fn resolve(&self, file_id: FileId, path: &RelativePath) -> Option<FileId>;
    fn debug_path(&self, _1file_id: FileId) -> Option<std::path::PathBuf> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct FileResolverImp {
    inner: Arc<FileResolver>,
}

impl PartialEq for FileResolverImp {
    fn eq(&self, other: &FileResolverImp) -> bool {
        self.inner() == other.inner()
    }
}

impl Eq for FileResolverImp {}

impl Hash for FileResolverImp {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.inner().hash(hasher);
    }
}

impl FileResolverImp {
    pub fn new(inner: Arc<FileResolver>) -> FileResolverImp {
        FileResolverImp { inner }
    }
    pub fn file_stem(&self, file_id: FileId) -> String {
        self.inner.file_stem(file_id)
    }
    pub fn resolve(&self, file_id: FileId, path: &RelativePath) -> Option<FileId> {
        self.inner.resolve(file_id, path)
    }
    pub fn debug_path(&self, file_id: FileId) -> Option<std::path::PathBuf> {
        self.inner.debug_path(file_id)
    }
    fn inner(&self) -> *const FileResolver {
        &*self.inner
    }
}

impl Default for FileResolverImp {
    fn default() -> FileResolverImp {
        #[derive(Debug)]
        struct DummyResolver;
        impl FileResolver for DummyResolver {
            fn file_stem(&self, _file_: FileId) -> String {
                panic!("file resolver not set")
            }
            fn resolve(
                &self,
                _file_id: FileId,
                _path: &::relative_path::RelativePath,
            ) -> Option<FileId> {
                panic!("file resolver not set")
            }
        }
        FileResolverImp {
            inner: Arc::new(DummyResolver),
        }
    }
}
