use std::iter;

use base_db::{
    span::{HirFileId, HirFileIdRepr, MacroFile, SyntaxContextId},
    FileRange,
};
use either::Either;
use syntax::{AstNode, SyntaxNode, SyntaxToken, TextRange};

use crate::{db, ExpansionInfo, HirFileIdExt as _};

// FIXME: Make an InRealFile wrapper
/// `InFile<T>` stores a value of `T` inside a particular file/syntax tree.
///
/// Typical usages are:
///
/// * `InFile<SyntaxNode>` -- syntax node in a file
/// * `InFile<ast::FnDef>` -- ast node in a file
/// * `InFile<TextSize>` -- offset in a file
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct InFile<T> {
    pub file_id: HirFileId,
    pub value: T,
}

impl<T> InFile<T> {
    pub fn new(file_id: HirFileId, value: T) -> InFile<T> {
        InFile { file_id, value }
    }

    pub fn with_value<U>(&self, value: U) -> InFile<U> {
        InFile::new(self.file_id, value)
    }

    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> InFile<U> {
        InFile::new(self.file_id, f(self.value))
    }

    pub fn as_ref(&self) -> InFile<&T> {
        self.with_value(&self.value)
    }

    pub fn file_syntax(&self, db: &dyn db::ExpandDatabase) -> SyntaxNode {
        db.parse_or_expand(self.file_id)
    }
}

impl<T: Clone> InFile<&T> {
    pub fn cloned(&self) -> InFile<T> {
        self.with_value(self.value.clone())
    }
}

impl<T> InFile<Option<T>> {
    pub fn transpose(self) -> Option<InFile<T>> {
        let value = self.value?;
        Some(InFile::new(self.file_id, value))
    }
}

impl<L, R> InFile<Either<L, R>> {
    pub fn transpose(self) -> Either<InFile<L>, InFile<R>> {
        match self.value {
            Either::Left(l) => Either::Left(InFile::new(self.file_id, l)),
            Either::Right(r) => Either::Right(InFile::new(self.file_id, r)),
        }
    }
}

impl InFile<&SyntaxNode> {
    pub fn ancestors_with_macros(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> impl Iterator<Item = InFile<SyntaxNode>> + Clone + '_ {
        iter::successors(Some(self.cloned()), move |node| match node.value.parent() {
            Some(parent) => Some(node.with_value(parent)),
            None => node.file_id.call_node(db),
        })
    }

    /// Skips the attributed item that caused the macro invocation we are climbing up
    pub fn ancestors_with_macros_skip_attr_item(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> impl Iterator<Item = InFile<SyntaxNode>> + '_ {
        let succ = move |node: &InFile<SyntaxNode>| match node.value.parent() {
            Some(parent) => Some(node.with_value(parent)),
            None => {
                let parent_node = node.file_id.call_node(db)?;
                if node.file_id.is_attr_macro(db) {
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
                    ExpansionInfo::new(db, mac_file).map_node_range_up(db, self.value.text_range())
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
    pub fn original_file_range_full(self, db: &dyn db::ExpandDatabase) -> FileRange {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => FileRange { file_id, range: self.value.text_range() },
            HirFileIdRepr::MacroFile(mac_file) => {
                if let Some((res, ctxt)) =
                    ExpansionInfo::new(db, mac_file).map_node_range_up(db, self.value.text_range())
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
                ExpansionInfo::new(db, mac_file).map_node_range_up(db, self.value.text_range())
            }
        }
    }

    pub fn original_syntax_node(self, db: &dyn db::ExpandDatabase) -> Option<InFile<SyntaxNode>> {
        // This kind of upmapping can only be achieved in attribute expanded files,
        // as we don't have node inputs otherwise and therefore can't find an `N` node in the input
        let Some(file_id) = self.file_id.macro_file() else {
            return Some(self.map(Clone::clone));
        };
        if !self.file_id.is_attr_macro(db) {
            return None;
        }

        let (FileRange { file_id, range }, ctx) =
            ExpansionInfo::new(db, file_id).map_node_range_up(db, self.value.text_range())?;

        // FIXME: Figure out an API that makes proper use of ctx, this only exists to
        // keep pre-token map rewrite behaviour.
        if !ctx.is_root() {
            return None;
        }

        let anc = db.parse(file_id).syntax_node().covering_element(range);
        let kind = self.value.kind();
        // FIXME: This heuristic is brittle and with the right macro may select completely unrelated nodes?
        let value = anc.ancestors().find(|it| it.kind() == kind)?;
        Some(InFile::new(file_id.into(), value))
    }
}

impl InFile<SyntaxToken> {
    pub fn upmap_once(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> Option<InFile<smallvec::SmallVec<[TextRange; 1]>>> {
        Some(self.file_id.expansion_info(db)?.map_range_up_once(db, self.value.text_range()))
    }

    /// Falls back to the macro call range if the node cannot be mapped up fully.
    pub fn original_file_range(self, db: &dyn db::ExpandDatabase) -> FileRange {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => FileRange { file_id, range: self.value.text_range() },
            HirFileIdRepr::MacroFile(mac_file) => {
                if let Some(res) = self.original_file_range_opt(db) {
                    return res;
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
            HirFileIdRepr::MacroFile(_) => {
                let (range, ctxt) = ascend_range_up_macros(db, self.map(|it| it.text_range()));

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

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct InMacroFile<T> {
    pub file_id: MacroFile,
    pub value: T,
}

impl<T> From<InMacroFile<T>> for InFile<T> {
    fn from(macro_file: InMacroFile<T>) -> Self {
        InFile { file_id: macro_file.file_id.into(), value: macro_file.value }
    }
}

pub fn ascend_range_up_macros(
    db: &dyn db::ExpandDatabase,
    range: InFile<TextRange>,
) -> (FileRange, SyntaxContextId) {
    match range.file_id.repr() {
        HirFileIdRepr::FileId(file_id) => {
            (FileRange { file_id, range: range.value }, SyntaxContextId::ROOT)
        }
        HirFileIdRepr::MacroFile(m) => {
            ExpansionInfo::new(db, m).map_token_range_up(db, range.value)
        }
    }
}

impl<N: AstNode> InFile<N> {
    pub fn descendants<T: AstNode>(self) -> impl Iterator<Item = InFile<T>> {
        self.value.syntax().descendants().filter_map(T::cast).map(move |n| self.with_value(n))
    }

    // FIXME: this should return `Option<InFileNotHirFile<N>>`
    pub fn original_ast_node(self, db: &dyn db::ExpandDatabase) -> Option<InFile<N>> {
        // This kind of upmapping can only be achieved in attribute expanded files,
        // as we don't have node inputs otherwise and therefore can't find an `N` node in the input
        let Some(file_id) = self.file_id.macro_file() else {
            return Some(self);
        };
        if !self.file_id.is_attr_macro(db) {
            return None;
        }

        let (FileRange { file_id, range }, ctx) = ExpansionInfo::new(db, file_id)
            .map_node_range_up(db, self.value.syntax().text_range())?;

        // FIXME: Figure out an API that makes proper use of ctx, this only exists to
        // keep pre-token map rewrite behaviour.
        if !ctx.is_root() {
            return None;
        }

        // FIXME: This heuristic is brittle and with the right macro may select completely unrelated nodes?
        let anc = db.parse(file_id).syntax_node().covering_element(range);
        let value = anc.ancestors().find_map(N::cast)?;
        return Some(InFile::new(file_id.into(), value));
    }

    pub fn syntax(&self) -> InFile<&SyntaxNode> {
        self.with_value(self.value.syntax())
    }
}
