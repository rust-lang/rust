//! See `AssistContext`

use std::mem;

use algo::find_covering_element;
use base_db::{FileId, FileRange};
use hir::Semantics;
use ide_db::{
    label::Label,
    source_change::{SourceChange, SourceFileEdit},
    RootDatabase,
};
use syntax::{
    algo::{self, find_node_at_offset, SyntaxRewriter},
    AstNode, SourceFile, SyntaxElement, SyntaxKind, SyntaxToken, TextRange, TextSize,
    TokenAtOffset,
};
use text_edit::{TextEdit, TextEditBuilder};

use crate::{
    assist_config::{AssistConfig, SnippetCap},
    Assist, AssistId, AssistKind, GroupLabel, ResolvedAssist,
};

/// `AssistContext` allows to apply an assist or check if it could be applied.
///
/// Assists use a somewhat over-engineered approach, given the current needs.
/// The assists workflow consists of two phases. In the first phase, a user asks
/// for the list of available assists. In the second phase, the user picks a
/// particular assist and it gets applied.
///
/// There are two peculiarities here:
///
/// * first, we ideally avoid computing more things then necessary to answer "is
///   assist applicable" in the first phase.
/// * second, when we are applying assist, we don't have a guarantee that there
///   weren't any changes between the point when user asked for assists and when
///   they applied a particular assist. So, when applying assist, we need to do
///   all the checks from scratch.
///
/// To avoid repeating the same code twice for both "check" and "apply"
/// functions, we use an approach reminiscent of that of Django's function based
/// views dealing with forms. Each assist receives a runtime parameter,
/// `resolve`. It first check if an edit is applicable (potentially computing
/// info required to compute the actual edit). If it is applicable, and
/// `resolve` is `true`, it then computes the actual edit.
///
/// So, to implement the original assists workflow, we can first apply each edit
/// with `resolve = false`, and then applying the selected edit again, with
/// `resolve = true` this time.
///
/// Note, however, that we don't actually use such two-phase logic at the
/// moment, because the LSP API is pretty awkward in this place, and it's much
/// easier to just compute the edit eagerly :-)
pub(crate) struct AssistContext<'a> {
    pub(crate) config: &'a AssistConfig,
    pub(crate) sema: Semantics<'a, RootDatabase>,
    pub(crate) frange: FileRange,
    source_file: SourceFile,
}

impl<'a> AssistContext<'a> {
    pub(crate) fn new(
        sema: Semantics<'a, RootDatabase>,
        config: &'a AssistConfig,
        frange: FileRange,
    ) -> AssistContext<'a> {
        let source_file = sema.parse(frange.file_id);
        AssistContext { config, sema, frange, source_file }
    }

    pub(crate) fn db(&self) -> &RootDatabase {
        self.sema.db
    }

    pub(crate) fn source_file(&self) -> &SourceFile {
        &self.source_file
    }

    // NB, this ignores active selection.
    pub(crate) fn offset(&self) -> TextSize {
        self.frange.range.start()
    }

    pub(crate) fn token_at_offset(&self) -> TokenAtOffset<SyntaxToken> {
        self.source_file.syntax().token_at_offset(self.offset())
    }
    pub(crate) fn find_token_at_offset(&self, kind: SyntaxKind) -> Option<SyntaxToken> {
        self.token_at_offset().find(|it| it.kind() == kind)
    }
    pub(crate) fn find_node_at_offset<N: AstNode>(&self) -> Option<N> {
        find_node_at_offset(self.source_file.syntax(), self.offset())
    }
    pub(crate) fn find_node_at_offset_with_descend<N: AstNode>(&self) -> Option<N> {
        self.sema.find_node_at_offset_with_descend(self.source_file.syntax(), self.offset())
    }
    pub(crate) fn covering_element(&self) -> SyntaxElement {
        find_covering_element(self.source_file.syntax(), self.frange.range)
    }
    // FIXME: remove
    pub(crate) fn covering_node_for_range(&self, range: TextRange) -> SyntaxElement {
        find_covering_element(self.source_file.syntax(), range)
    }
}

pub(crate) struct Assists {
    resolve: bool,
    file: FileId,
    buf: Vec<(Assist, Option<SourceChange>)>,
    allowed: Option<Vec<AssistKind>>,
}

impl Assists {
    pub(crate) fn new_resolved(ctx: &AssistContext) -> Assists {
        Assists {
            resolve: true,
            file: ctx.frange.file_id,
            buf: Vec::new(),
            allowed: ctx.config.allowed.clone(),
        }
    }

    pub(crate) fn new_unresolved(ctx: &AssistContext) -> Assists {
        Assists {
            resolve: false,
            file: ctx.frange.file_id,
            buf: Vec::new(),
            allowed: ctx.config.allowed.clone(),
        }
    }

    pub(crate) fn finish_unresolved(self) -> Vec<Assist> {
        assert!(!self.resolve);
        self.finish()
            .into_iter()
            .map(|(label, edit)| {
                assert!(edit.is_none());
                label
            })
            .collect()
    }

    pub(crate) fn finish_resolved(self) -> Vec<ResolvedAssist> {
        assert!(self.resolve);
        self.finish()
            .into_iter()
            .map(|(label, edit)| ResolvedAssist { assist: label, source_change: edit.unwrap() })
            .collect()
    }

    pub(crate) fn add(
        &mut self,
        id: AssistId,
        label: impl Into<String>,
        target: TextRange,
        f: impl FnOnce(&mut AssistBuilder),
    ) -> Option<()> {
        if !self.is_allowed(&id) {
            return None;
        }
        let label = Label::new(label.into());
        let assist = Assist { id, label, group: None, target };
        self.add_impl(assist, f)
    }

    pub(crate) fn add_group(
        &mut self,
        group: &GroupLabel,
        id: AssistId,
        label: impl Into<String>,
        target: TextRange,
        f: impl FnOnce(&mut AssistBuilder),
    ) -> Option<()> {
        if !self.is_allowed(&id) {
            return None;
        }
        let label = Label::new(label.into());
        let assist = Assist { id, label, group: Some(group.clone()), target };
        self.add_impl(assist, f)
    }

    fn add_impl(&mut self, assist: Assist, f: impl FnOnce(&mut AssistBuilder)) -> Option<()> {
        let source_change = if self.resolve {
            let mut builder = AssistBuilder::new(self.file);
            f(&mut builder);
            Some(builder.finish())
        } else {
            None
        };

        self.buf.push((assist, source_change));
        Some(())
    }

    fn finish(mut self) -> Vec<(Assist, Option<SourceChange>)> {
        self.buf.sort_by_key(|(label, _edit)| label.target.len());
        self.buf
    }

    fn is_allowed(&self, id: &AssistId) -> bool {
        match &self.allowed {
            Some(allowed) => allowed.iter().any(|kind| kind.contains(id.1)),
            None => true,
        }
    }
}

pub(crate) struct AssistBuilder {
    edit: TextEditBuilder,
    file_id: FileId,
    is_snippet: bool,
    change: SourceChange,
}

impl AssistBuilder {
    pub(crate) fn new(file_id: FileId) -> AssistBuilder {
        AssistBuilder {
            edit: TextEdit::builder(),
            file_id,
            is_snippet: false,
            change: SourceChange::default(),
        }
    }

    pub(crate) fn edit_file(&mut self, file_id: FileId) {
        self.file_id = file_id;
    }

    fn commit(&mut self) {
        let edit = mem::take(&mut self.edit).finish();
        if !edit.is_empty() {
            let new_edit = SourceFileEdit { file_id: self.file_id, edit };
            assert!(!self.change.source_file_edits.iter().any(|it| it.file_id == new_edit.file_id));
            self.change.source_file_edits.push(new_edit);
        }
    }

    /// Remove specified `range` of text.
    pub(crate) fn delete(&mut self, range: TextRange) {
        self.edit.delete(range)
    }
    /// Append specified `text` at the given `offset`
    pub(crate) fn insert(&mut self, offset: TextSize, text: impl Into<String>) {
        self.edit.insert(offset, text.into())
    }
    /// Append specified `snippet` at the given `offset`
    pub(crate) fn insert_snippet(
        &mut self,
        _cap: SnippetCap,
        offset: TextSize,
        snippet: impl Into<String>,
    ) {
        self.is_snippet = true;
        self.insert(offset, snippet);
    }
    /// Replaces specified `range` of text with a given string.
    pub(crate) fn replace(&mut self, range: TextRange, replace_with: impl Into<String>) {
        self.edit.replace(range, replace_with.into())
    }
    /// Replaces specified `range` of text with a given `snippet`.
    pub(crate) fn replace_snippet(
        &mut self,
        _cap: SnippetCap,
        range: TextRange,
        snippet: impl Into<String>,
    ) {
        self.is_snippet = true;
        self.replace(range, snippet);
    }
    pub(crate) fn replace_ast<N: AstNode>(&mut self, old: N, new: N) {
        algo::diff(old.syntax(), new.syntax()).into_text_edit(&mut self.edit)
    }
    pub(crate) fn rewrite(&mut self, rewriter: SyntaxRewriter) {
        let node = rewriter.rewrite_root().unwrap();
        let new = rewriter.rewrite(&node);
        algo::diff(&node, &new).into_text_edit(&mut self.edit);
    }

    // FIXME: kill this API
    /// Get access to the raw `TextEditBuilder`.
    pub(crate) fn text_edit_builder(&mut self) -> &mut TextEditBuilder {
        &mut self.edit
    }

    fn finish(mut self) -> SourceChange {
        self.commit();
        let mut change = mem::take(&mut self.change);
        if self.is_snippet {
            change.is_snippet = true;
        }
        change
    }
}
