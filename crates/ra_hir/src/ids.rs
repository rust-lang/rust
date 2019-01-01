use crate::{FileId, MacroCallId};

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
pub enum MFileId {
    File(FileId),
    Macro(MacroCallId),
}

impl From<FileId> for MFileId {
    fn from(file_id: FileId) -> MFileId {
        MFileId::File(file_id)
    }
}
