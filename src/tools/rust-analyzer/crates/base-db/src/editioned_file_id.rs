//! Defines [`EditionedFileId`], an interned wrapper around [`span::EditionedFileId`] that
//! is interned (so queries can take it) and stores only the underlying `span::EditionedFileId`.

use std::hash::Hash;

use salsa::Database;
use span::Edition;
use vfs::FileId;

#[salsa::interned(debug, constructor = from_span_file_id, no_lifetime)]
#[derive(PartialOrd, Ord)]
pub struct EditionedFileId {
    field: span::EditionedFileId,
}

impl EditionedFileId {
    #[inline]
    pub fn new(db: &dyn Database, file_id: FileId, edition: Edition) -> Self {
        Self::from_span_file_id(db, span::EditionedFileId::new(file_id, edition))
    }

    #[inline]
    pub fn current_edition(db: &dyn Database, file_id: FileId) -> Self {
        Self::from_span_file_id(db, span::EditionedFileId::current_edition(file_id))
    }

    #[inline]
    pub fn file_id(self, db: &dyn Database) -> vfs::FileId {
        self.field(db).file_id()
    }

    #[inline]
    pub fn span_file_id(self, db: &dyn Database) -> span::EditionedFileId {
        self.field(db)
    }

    #[inline]
    pub fn unpack(self, db: &dyn Database) -> (vfs::FileId, span::Edition) {
        self.field(db).unpack()
    }

    #[inline]
    pub fn edition(self, db: &dyn Database) -> Edition {
        self.field(db).edition()
    }
}
