//! Defines [`EditionedFileId`], an interned wrapper around [`span::EditionedFileId`] that
//! is interned (so queries can take it) and stores only the underlying `span::EditionedFileId`.

use std::hash::Hash;

use salsa::Database;
use span::Edition;
use syntax::{SyntaxError, ast};
use vfs::FileId;

use crate::SourceDatabase;

#[salsa::interned(debug, constructor = from_span_file_id, no_lifetime)]
#[derive(PartialOrd, Ord)]
pub struct EditionedFileId {
    field: span::EditionedFileId,
}

// Currently does not work due to a salsa bug
// #[salsa::tracked]
// impl EditionedFileId {
//     #[salsa::tracked(lru = 128)]
//     pub fn parse(self, db: &dyn SourceDatabase) -> syntax::Parse<ast::SourceFile> {
//         let _p = tracing::info_span!("parse", ?self).entered();
//         let (file_id, edition) = self.unpack(db);
//         let text = db.file_text(file_id).text(db);
//         ast::SourceFile::parse(text, edition)
//     }

//     // firewall query
//     #[salsa::tracked(returns(as_deref))]
//     pub fn parse_errors(self, db: &dyn SourceDatabase) -> Option<Box<[SyntaxError]>> {
//         let errors = self.parse(db).errors();
//         match &*errors {
//             [] => None,
//             [..] => Some(errors.into()),
//         }
//     }
// }

impl EditionedFileId {
    pub fn parse(self, db: &dyn SourceDatabase) -> syntax::Parse<ast::SourceFile> {
        #[salsa::tracked(lru = 128)]
        pub fn parse(
            db: &dyn SourceDatabase,
            file_id: EditionedFileId,
        ) -> syntax::Parse<ast::SourceFile> {
            let _p = tracing::info_span!("parse", ?file_id).entered();
            let (file_id, edition) = file_id.unpack(db);
            let text = db.file_text(file_id).text(db);
            ast::SourceFile::parse(text, edition)
        }
        parse(db, self)
    }

    // firewall query
    pub fn parse_errors(self, db: &dyn SourceDatabase) -> Option<&[SyntaxError]> {
        #[salsa::tracked(returns(as_deref))]
        pub fn parse_errors(
            db: &dyn SourceDatabase,
            file_id: EditionedFileId,
        ) -> Option<Box<[SyntaxError]>> {
            let errors = file_id.parse(db).errors();
            match &*errors {
                [] => None,
                [..] => Some(errors.into()),
            }
        }
        parse_errors(db, self)
    }
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
