//! ra_db defines basic database traits. Concrete DB is defined by ra_analysis.

extern crate ra_editor;
extern crate ra_syntax;
extern crate relative_path;
extern crate rustc_hash;
extern crate salsa;

mod syntax_ptr;
mod file_resolver;
mod input;
mod loc2id;

use std::sync::Arc;
use ra_editor::LineIndex;
use ra_syntax::SourceFileNode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Canceled;

pub type Cancelable<T> = Result<T, Canceled>;

impl std::fmt::Display for Canceled {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str("Canceled")
    }
}

impl std::error::Error for Canceled {}

pub use crate::{
    syntax_ptr::LocalSyntaxPtr,
    file_resolver::{FileResolver, FileResolverImp},
    input::{
        FilesDatabase, FileId, CrateId, SourceRoot, SourceRootId, CrateGraph, WORKSPACE,
        FileTextQuery, FileSourceRootQuery, SourceRootQuery, LibrariesQuery, CrateGraphQuery,
    },
    loc2id::{LocationIntener, NumericId},
};

#[macro_export]
macro_rules! impl_numeric_id {
    ($id:ident) => {
        impl $crate::NumericId for $id {
            fn from_u32(id: u32) -> Self {
                $id(id)
            }
            fn to_u32(self) -> u32 {
                self.0
            }
        }
    };
}

pub trait BaseDatabase: salsa::Database {
    fn check_canceled(&self) -> Cancelable<()> {
        if self.salsa_runtime().is_current_revision_canceled() {
            Err(Canceled)
        } else {
            Ok(())
        }
    }
}

salsa::query_group! {
    pub trait SyntaxDatabase: crate::input::FilesDatabase + BaseDatabase {
        fn source_file(file_id: FileId) -> SourceFileNode {
            type SourceFileQuery;
        }
        fn file_lines(file_id: FileId) -> Arc<LineIndex> {
            type FileLinesQuery;
        }
    }
}

fn source_file(db: &impl SyntaxDatabase, file_id: FileId) -> SourceFileNode {
    let text = db.file_text(file_id);
    SourceFileNode::parse(&*text)
}
fn file_lines(db: &impl SyntaxDatabase, file_id: FileId) -> Arc<LineIndex> {
    let text = db.file_text(file_id);
    Arc::new(LineIndex::new(&*text))
}
