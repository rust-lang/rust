//! Things to wrap other things in file ids.
use std::iter;

use either::Either;
use span::{FileId, FileRange, HirFileId, HirFileIdRepr, MacroFileId, SyntaxContextId};
use syntax::{AstNode, SyntaxNode, SyntaxToken, TextRange, TextSize};

use crate::{db, map_node_range_up, span_for_offset, MacroFileIdExt};

/// `InFile<T>` stores a value of `T` inside a particular file/syntax tree.
///
/// Typical usages are:
///
/// * `InFile<SyntaxNode>` -- syntax node in a file
/// * `InFile<ast::FnDef>` -- ast node in a file
/// * `InFile<TextSize>` -- offset in a file
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct InFileWrapper<FileKind, T> {
    pub file_id: FileKind,
    pub value: T,
}
pub type InFile<T> = InFileWrapper<HirFileId, T>;
pub type InMacroFile<T> = InFileWrapper<MacroFileId, T>;
pub type InRealFile<T> = InFileWrapper<FileId, T>;

impl<FileKind, T> InFileWrapper<FileKind, T> {
    pub fn new(file_id: FileKind, value: T) -> Self {
        Self { file_id, value }
    }

    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> InFileWrapper<FileKind, U> {
        InFileWrapper::new(self.file_id, f(self.value))
    }
}

impl<FileKind: Copy, T> InFileWrapper<FileKind, T> {
    pub fn with_value<U>(&self, value: U) -> InFileWrapper<FileKind, U> {
        InFileWrapper::new(self.file_id, value)
    }

    pub fn as_ref(&self) -> InFileWrapper<FileKind, &T> {
        self.with_value(&self.value)
    }
}

impl<FileKind: Copy, T: Clone> InFileWrapper<FileKind, &T> {
    pub fn cloned(&self) -> InFileWrapper<FileKind, T> {
        self.with_value(self.value.clone())
    }
}

impl<T> From<InMacroFile<T>> for InFile<T> {
    fn from(InMacroFile { file_id, value }: InMacroFile<T>) -> Self {
        InFile { file_id: file_id.into(), value }
    }
}

impl<T> From<InRealFile<T>> for InFile<T> {
    fn from(InRealFile { file_id, value }: InRealFile<T>) -> Self {
        InFile { file_id: file_id.into(), value }
    }
}

// region:transpose impls

impl<FileKind, T> InFileWrapper<FileKind, Option<T>> {
    pub fn transpose(self) -> Option<InFileWrapper<FileKind, T>> {
        Some(InFileWrapper::new(self.file_id, self.value?))
    }
}

impl<FileKind, L, R> InFileWrapper<FileKind, Either<L, R>> {
    pub fn transpose(self) -> Either<InFileWrapper<FileKind, L>, InFileWrapper<FileKind, R>> {
        match self.value {
            Either::Left(l) => Either::Left(InFileWrapper::new(self.file_id, l)),
            Either::Right(r) => Either::Right(InFileWrapper::new(self.file_id, r)),
        }
    }
}

// endregion:transpose impls

trait FileIdToSyntax: Copy {
    fn file_syntax(self, db: &dyn db::ExpandDatabase) -> SyntaxNode;
}

impl FileIdToSyntax for FileId {
    fn file_syntax(self, db: &dyn db::ExpandDatabase) -> SyntaxNode {
        db.parse(self).syntax_node()
    }
}
impl FileIdToSyntax for MacroFileId {
    fn file_syntax(self, db: &dyn db::ExpandDatabase) -> SyntaxNode {
        db.parse_macro_expansion(self).value.0.syntax_node()
    }
}
impl FileIdToSyntax for HirFileId {
    fn file_syntax(self, db: &dyn db::ExpandDatabase) -> SyntaxNode {
        db.parse_or_expand(self)
    }
}

#[allow(private_bounds)]
impl<FileId: FileIdToSyntax, T> InFileWrapper<FileId, T> {
    pub fn file_syntax(&self, db: &dyn db::ExpandDatabase) -> SyntaxNode {
        FileIdToSyntax::file_syntax(self.file_id, db)
    }
}

impl<FileId: Copy, N: AstNode> InFileWrapper<FileId, N> {
    pub fn syntax(&self) -> InFileWrapper<FileId, &SyntaxNode> {
        self.with_value(self.value.syntax())
    }
}

// region:specific impls

impl InFile<&SyntaxNode> {
    /// Skips the attributed item that caused the macro invocation we are climbing up
    pub fn ancestors_with_macros_skip_attr_item(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> impl Iterator<Item = InFile<SyntaxNode>> + '_ {
        let succ = move |node: &InFile<SyntaxNode>| match node.value.parent() {
            Some(parent) => Some(node.with_value(parent)),
            None => {
                let macro_file_id = node.file_id.macro_file()?;
                let parent_node = macro_file_id.call_node(db);
                if macro_file_id.is_attr_macro(db) {
                    // macro call was an attributed item, skip it
                    // FIXME: does this fail if this is a direct expansion of another macro?
                    parent_node.map(|node| node.parent()).transpose()
                } else {
                    Some(parent_node)
                }
            }
        };
        iter::successors(succ(&self.cloned()), succ)
    }

    /// Falls back to the macro call range if the node cannot be mapped up fully.
    ///
    /// For attributes and derives, this will point back to the attribute only.
    /// For the entire item use [`InFile::original_file_range_full`].
    pub fn original_file_range(self, db: &dyn db::ExpandDatabase) -> FileRange {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => FileRange { file_id, range: self.value.text_range() },
            HirFileIdRepr::MacroFile(mac_file) => {
                if let Some((res, ctxt)) =
                    map_node_range_up(db, &db.expansion_span_map(mac_file), self.value.text_range())
                {
                    // FIXME: Figure out an API that makes proper use of ctx, this only exists to
                    // keep pre-token map rewrite behaviour.
                    if ctxt.is_root() {
                        return res;
                    }
                }
                // Fall back to whole macro call.
                let loc = db.lookup_intern_macro_call(mac_file.macro_call_id);
                loc.kind.original_call_range(db)
            }
        }
    }

    /// Falls back to the macro call range if the node cannot be mapped up fully.
    pub fn original_file_range_with_macro_call_body(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> FileRange {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => FileRange { file_id, range: self.value.text_range() },
            HirFileIdRepr::MacroFile(mac_file) => {
                if let Some((res, ctxt)) =
                    map_node_range_up(db, &db.expansion_span_map(mac_file), self.value.text_range())
                {
                    // FIXME: Figure out an API that makes proper use of ctx, this only exists to
                    // keep pre-token map rewrite behaviour.
                    if ctxt.is_root() {
                        return res;
                    }
                }
                // Fall back to whole macro call.
                let loc = db.lookup_intern_macro_call(mac_file.macro_call_id);
                loc.kind.original_call_range_with_body(db)
            }
        }
    }

    /// Attempts to map the syntax node back up its macro calls.
    pub fn original_file_range_opt(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> Option<(FileRange, SyntaxContextId)> {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => {
                Some((FileRange { file_id, range: self.value.text_range() }, SyntaxContextId::ROOT))
            }
            HirFileIdRepr::MacroFile(mac_file) => {
                map_node_range_up(db, &db.expansion_span_map(mac_file), self.value.text_range())
            }
        }
    }

    pub fn original_syntax_node(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> Option<InRealFile<SyntaxNode>> {
        // This kind of upmapping can only be achieved in attribute expanded files,
        // as we don't have node inputs otherwise and therefore can't find an `N` node in the input
        let file_id = match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => {
                return Some(InRealFile { file_id, value: self.value.clone() })
            }
            HirFileIdRepr::MacroFile(m) => m,
        };
        if !file_id.is_attr_macro(db) {
            return None;
        }

        let (FileRange { file_id, range }, ctx) =
            map_node_range_up(db, &db.expansion_span_map(file_id), self.value.text_range())?;

        // FIXME: Figure out an API that makes proper use of ctx, this only exists to
        // keep pre-token map rewrite behaviour.
        if !ctx.is_root() {
            return None;
        }

        let anc = db.parse(file_id).syntax_node().covering_element(range);
        let kind = self.value.kind();
        // FIXME: This heuristic is brittle and with the right macro may select completely unrelated nodes?
        let value = anc.ancestors().find(|it| it.kind() == kind)?;
        Some(InRealFile::new(file_id, value))
    }
}

impl InMacroFile<SyntaxToken> {
    pub fn upmap_once(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> InFile<smallvec::SmallVec<[TextRange; 1]>> {
        self.file_id.expansion_info(db).map_range_up_once(db, self.value.text_range())
    }
}

impl InFile<SyntaxToken> {
    /// Falls back to the macro call range if the node cannot be mapped up fully.
    pub fn original_file_range(self, db: &dyn db::ExpandDatabase) -> FileRange {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => FileRange { file_id, range: self.value.text_range() },
            HirFileIdRepr::MacroFile(mac_file) => {
                let (range, ctxt) = span_for_offset(
                    db,
                    &db.expansion_span_map(mac_file),
                    self.value.text_range().start(),
                );

                // FIXME: Figure out an API that makes proper use of ctx, this only exists to
                // keep pre-token map rewrite behaviour.
                if ctxt.is_root() {
                    return range;
                }

                // Fall back to whole macro call.
                let loc = db.lookup_intern_macro_call(mac_file.macro_call_id);
                loc.kind.original_call_range(db)
            }
        }
    }

    /// Attempts to map the syntax node back up its macro calls.
    pub fn original_file_range_opt(self, db: &dyn db::ExpandDatabase) -> Option<FileRange> {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => {
                Some(FileRange { file_id, range: self.value.text_range() })
            }
            HirFileIdRepr::MacroFile(mac_file) => {
                let (range, ctxt) = span_for_offset(
                    db,
                    &db.expansion_span_map(mac_file),
                    self.value.text_range().start(),
                );

                // FIXME: Figure out an API that makes proper use of ctx, this only exists to
                // keep pre-token map rewrite behaviour.
                if ctxt.is_root() {
                    Some(range)
                } else {
                    None
                }
            }
        }
    }
}

impl InMacroFile<TextSize> {
    pub fn original_file_range(self, db: &dyn db::ExpandDatabase) -> (FileRange, SyntaxContextId) {
        span_for_offset(db, &db.expansion_span_map(self.file_id), self.value)
    }
}

impl InFile<TextRange> {
    pub fn original_node_file_range(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> (FileRange, SyntaxContextId) {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => {
                (FileRange { file_id, range: self.value }, SyntaxContextId::ROOT)
            }
            HirFileIdRepr::MacroFile(mac_file) => {
                match map_node_range_up(db, &db.expansion_span_map(mac_file), self.value) {
                    Some(it) => it,
                    None => {
                        let loc = db.lookup_intern_macro_call(mac_file.macro_call_id);
                        (loc.kind.original_call_range(db), SyntaxContextId::ROOT)
                    }
                }
            }
        }
    }

    pub fn original_node_file_range_rooted(self, db: &dyn db::ExpandDatabase) -> FileRange {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => FileRange { file_id, range: self.value },
            HirFileIdRepr::MacroFile(mac_file) => {
                match map_node_range_up(db, &db.expansion_span_map(mac_file), self.value) {
                    Some((it, SyntaxContextId::ROOT)) => it,
                    _ => {
                        let loc = db.lookup_intern_macro_call(mac_file.macro_call_id);
                        loc.kind.original_call_range(db)
                    }
                }
            }
        }
    }

    pub fn original_node_file_range_opt(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> Option<(FileRange, SyntaxContextId)> {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => {
                Some((FileRange { file_id, range: self.value }, SyntaxContextId::ROOT))
            }
            HirFileIdRepr::MacroFile(mac_file) => {
                map_node_range_up(db, &db.expansion_span_map(mac_file), self.value)
            }
        }
    }
}

impl<N: AstNode> InFile<N> {
    pub fn original_ast_node_rooted(self, db: &dyn db::ExpandDatabase) -> Option<InRealFile<N>> {
        // This kind of upmapping can only be achieved in attribute expanded files,
        // as we don't have node inputs otherwise and therefore can't find an `N` node in the input
        let file_id = match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => {
                return Some(InRealFile { file_id, value: self.value })
            }
            HirFileIdRepr::MacroFile(m) => m,
        };
        if !file_id.is_attr_macro(db) {
            return None;
        }

        let (FileRange { file_id, range }, ctx) = map_node_range_up(
            db,
            &db.expansion_span_map(file_id),
            self.value.syntax().text_range(),
        )?;

        // FIXME: Figure out an API that makes proper use of ctx, this only exists to
        // keep pre-token map rewrite behaviour.
        if !ctx.is_root() {
            return None;
        }

        // FIXME: This heuristic is brittle and with the right macro may select completely unrelated nodes?
        let anc = db.parse(file_id).syntax_node().covering_element(range);
        let value = anc.ancestors().find_map(N::cast)?;
        Some(InRealFile::new(file_id, value))
    }
}
