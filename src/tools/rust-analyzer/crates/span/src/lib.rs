//! File and span related types.
use std::fmt::{self, Write};

mod ast_id;
mod hygiene;
mod map;

pub use self::{
    ast_id::{
        AstIdMap, AstIdNode, ErasedFileAstId, FIXUP_ERASED_FILE_AST_ID_MARKER, FileAstId,
        ROOT_ERASED_FILE_AST_ID,
    },
    hygiene::{SyntaxContext, Transparency},
    map::{RealSpanMap, SpanMap},
};

pub use syntax::Edition;
pub use text_size::{TextRange, TextSize};
pub use vfs::FileId;

pub type Span = SpanData<SyntaxContext>;

impl Span {
    pub fn cover(self, other: Span) -> Span {
        if self.anchor != other.anchor {
            return self;
        }
        let range = self.range.cover(other.range);
        Span { range, ..self }
    }
}

/// Spans represent a region of code, used by the IDE to be able link macro inputs and outputs
/// together. Positions in spans are relative to some [`SpanAnchor`] to make them more incremental
/// friendly.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanData<Ctx> {
    /// The text range of this span, relative to the anchor.
    /// We need the anchor for incrementality, as storing absolute ranges will require
    /// recomputation on every change in a file at all times.
    pub range: TextRange,
    /// The anchor this span is relative to.
    pub anchor: SpanAnchor,
    /// The syntax context of the span.
    pub ctx: Ctx,
}

impl<Ctx: fmt::Debug> fmt::Debug for SpanData<Ctx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            fmt::Debug::fmt(&self.anchor.file_id.file_id().index(), f)?;
            f.write_char(':')?;
            write!(f, "{:#?}", self.anchor.ast_id)?;
            f.write_char('@')?;
            fmt::Debug::fmt(&self.range, f)?;
            f.write_char('#')?;
            self.ctx.fmt(f)
        } else {
            f.debug_struct("SpanData")
                .field("range", &self.range)
                .field("anchor", &self.anchor)
                .field("ctx", &self.ctx)
                .finish()
        }
    }
}

impl<Ctx: Copy> SpanData<Ctx> {
    pub fn eq_ignoring_ctx(self, other: Self) -> bool {
        self.anchor == other.anchor && self.range == other.range
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.anchor.file_id.file_id().index(), f)?;
        f.write_char(':')?;
        write!(f, "{:#?}", self.anchor.ast_id)?;
        f.write_char('@')?;
        fmt::Debug::fmt(&self.range, f)?;
        f.write_char('#')?;
        self.ctx.fmt(f)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct SpanAnchor {
    pub file_id: EditionedFileId,
    pub ast_id: ErasedFileAstId,
}

impl fmt::Debug for SpanAnchor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SpanAnchor").field(&self.file_id).field(&self.ast_id).finish()
    }
}

/// A [`FileId`] and [`Edition`] bundled up together.
/// The MSB is reserved for `HirFileId` encoding, more upper bits are used to then encode the edition.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EditionedFileId(u32);

impl fmt::Debug for EditionedFileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("EditionedFileId")
            .field(&self.file_id().index())
            .field(&self.edition())
            .finish()
    }
}

impl From<EditionedFileId> for FileId {
    fn from(value: EditionedFileId) -> Self {
        value.file_id()
    }
}

const _: () = assert!(
    EditionedFileId::RESERVED_HIGH_BITS
        + EditionedFileId::EDITION_BITS
        + EditionedFileId::FILE_ID_BITS
        == u32::BITS
);
const _: () = assert!(
    EditionedFileId::RESERVED_MASK ^ EditionedFileId::EDITION_MASK ^ EditionedFileId::FILE_ID_MASK
        == 0xFFFF_FFFF
);

impl EditionedFileId {
    pub const RESERVED_MASK: u32 = 0x8000_0000;
    pub const EDITION_MASK: u32 = 0x7F80_0000;
    pub const FILE_ID_MASK: u32 = 0x007F_FFFF;

    pub const MAX_FILE_ID: u32 = Self::FILE_ID_MASK;

    pub const RESERVED_HIGH_BITS: u32 = Self::RESERVED_MASK.count_ones();
    pub const FILE_ID_BITS: u32 = Self::FILE_ID_MASK.count_ones();
    pub const EDITION_BITS: u32 = Self::EDITION_MASK.count_ones();

    pub const fn current_edition(file_id: FileId) -> Self {
        Self::new(file_id, Edition::CURRENT)
    }

    pub const fn new(file_id: FileId, edition: Edition) -> Self {
        let file_id = file_id.index();
        let edition = edition as u32;
        assert!(file_id <= Self::MAX_FILE_ID);
        Self(file_id | (edition << Self::FILE_ID_BITS))
    }

    pub fn from_raw(u32: u32) -> Self {
        assert!(u32 & Self::RESERVED_MASK == 0);
        assert!((u32 & Self::EDITION_MASK) >> Self::FILE_ID_BITS <= Edition::LATEST as u32);
        Self(u32)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }

    pub const fn file_id(self) -> FileId {
        FileId::from_raw(self.0 & Self::FILE_ID_MASK)
    }

    pub const fn unpack(self) -> (FileId, Edition) {
        (self.file_id(), self.edition())
    }

    pub const fn edition(self) -> Edition {
        let edition = (self.0 & Self::EDITION_MASK) >> Self::FILE_ID_BITS;
        debug_assert!(edition <= Edition::LATEST as u32);
        unsafe { std::mem::transmute(edition as u8) }
    }
}

#[cfg(not(feature = "salsa"))]
mod salsa {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct Id(u32);
}

/// Input to the analyzer is a set of files, where each file is identified by
/// `FileId` and contains source code. However, another source of source code in
/// Rust are macros: each macro can be thought of as producing a "temporary
/// file". To assign an id to such a file, we use the id of the macro call that
/// produced the file. So, a `HirFileId` is either a `FileId` (source code
/// written by user), or a `MacroCallId` (source code produced by macro).
///
/// What is a `MacroCallId`? Simplifying, it's a `HirFileId` of a file
/// containing the call plus the offset of the macro call in the file. Note that
/// this is a recursive definition! However, the size_of of `HirFileId` is
/// finite (because everything bottoms out at the real `FileId`) and small
/// (`MacroCallId` uses the location interning. You can check details here:
/// <https://en.wikipedia.org/wiki/String_interning>).
///
/// Internally this holds a `salsa::Id`, but we cannot use this definition here
/// as it references things from base-db and hir-expand.
// FIXME: Give this a better fitting name
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HirFileId(pub salsa::Id);

/// `MacroCallId` identifies a particular macro invocation, like
/// `println!("Hello, {}", world)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MacroCallId(pub salsa::Id);
