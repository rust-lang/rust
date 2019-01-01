use crate::{FileId, MacroCallId, HirDatabase};

use ra_syntax::SourceFileNode;

/// hir makes a heavy use of ids: integer (u32) handlers to various things. You
/// can think of id as a pointer (but without a lifetime) or a file descriptor
/// (but for hir objects).
///
/// This module defines a bunch of ids we are using. The most important ones are
/// probably `HirFileId` and `DefId`.

/// Input to the analyzer is a set of file, where each file is indetified by
/// `FileId` and contains source code. However, another source of source code in
/// Rust are macros: each macro can be thought of as producing a "temporary
/// file". To assign id to such file, we use the id of a macro call that
/// produced the file. So, a `HirFileId` is either a `FileId` (source code
/// written by user), or a `MacroCallId` (source code produced by macro).
///
/// What is a `MacroCallId`? Simplifying, it's a `HirFileId` of a file containin
/// the call plus the offset of the macro call in the file. Note that this is a
/// recursive definition! Nethetheless, size_of of `HirFileId` is finite
/// (because everything bottoms out at the real `FileId`) and small
/// (`MacroCallId` uses location interner).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HirFileId(HirFileIdRepr);

impl HirFileId {
    pub(crate) fn original_file_id(self, db: &impl HirDatabase) -> FileId {
        match self.0 {
            HirFileIdRepr::File(file_id) => file_id,
            HirFileIdRepr::Macro(macro_call_id) => {
                let loc = macro_call_id.loc(db);
                loc.source_item_id.file_id.original_file_id(db)
            }
        }
    }

    pub(crate) fn as_original_file(self) -> FileId {
        match self.0 {
            HirFileIdRepr::File(file_id) => file_id,
            HirFileIdRepr::Macro(_r) => panic!("macro generated file: {:?}", self),
        }
    }
    pub(crate) fn source_file_query(db: &impl HirDatabase, file_id: HirFileId) -> SourceFileNode {
        match file_id.0 {
            HirFileIdRepr::File(file_id) => db.source_file(file_id),
            HirFileIdRepr::Macro(m) => {
                if let Some(exp) = db.expand_macro_invocation(m) {
                    return exp.file();
                }
                // returning an empty string looks fishy...
                SourceFileNode::parse("")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HirFileIdRepr {
    File(FileId),
    Macro(MacroCallId),
}

impl From<FileId> for HirFileId {
    fn from(file_id: FileId) -> HirFileId {
        HirFileId(HirFileIdRepr::File(file_id))
    }
}

impl From<MacroCallId> for HirFileId {
    fn from(macro_call_id: MacroCallId) -> HirFileId {
        HirFileId(HirFileIdRepr::Macro(macro_call_id))
    }
}
