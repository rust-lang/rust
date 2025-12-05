//! See [`AssistContext`].

use hir::{EditionedFileId, FileRange, Semantics};
use ide_db::{FileId, RootDatabase, label::Label};
use syntax::Edition;
use syntax::{
    AstNode, AstToken, Direction, SourceFile, SyntaxElement, SyntaxKind, SyntaxToken, TextRange,
    TextSize, TokenAtOffset,
    algo::{self, find_node_at_offset, find_node_at_range},
};

use crate::{
    Assist, AssistId, AssistKind, AssistResolveStrategy, GroupLabel, assist_config::AssistConfig,
};

pub(crate) use ide_db::source_change::{SourceChangeBuilder, TreeMutator};

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
    frange: FileRange,
    trimmed_range: TextRange,
    source_file: SourceFile,
    // We cache this here to speed up things slightly
    token_at_offset: TokenAtOffset<SyntaxToken>,
    // We cache this here to speed up things slightly
    covering_element: SyntaxElement,
}

impl<'a> AssistContext<'a> {
    pub(crate) fn new(
        sema: Semantics<'a, RootDatabase>,
        config: &'a AssistConfig,
        frange: FileRange,
    ) -> AssistContext<'a> {
        let source_file = sema.parse(frange.file_id);

        let start = frange.range.start();
        let end = frange.range.end();
        let left = source_file.syntax().token_at_offset(start);
        let right = source_file.syntax().token_at_offset(end);
        let left =
            left.right_biased().and_then(|t| algo::skip_whitespace_token(t, Direction::Next));
        let right =
            right.left_biased().and_then(|t| algo::skip_whitespace_token(t, Direction::Prev));
        let left = left.map(|t| t.text_range().start().clamp(start, end));
        let right = right.map(|t| t.text_range().end().clamp(start, end));

        let trimmed_range = match (left, right) {
            (Some(left), Some(right)) if left <= right => TextRange::new(left, right),
            // Selection solely consists of whitespace so just fall back to the original
            _ => frange.range,
        };
        let token_at_offset = source_file.syntax().token_at_offset(frange.range.start());
        let covering_element = source_file.syntax().covering_element(trimmed_range);

        AssistContext {
            config,
            sema,
            frange,
            source_file,
            trimmed_range,
            token_at_offset,
            covering_element,
        }
    }

    pub(crate) fn db(&self) -> &'a RootDatabase {
        self.sema.db
    }

    // NB, this ignores active selection.
    pub(crate) fn offset(&self) -> TextSize {
        self.frange.range.start()
    }

    pub(crate) fn vfs_file_id(&self) -> FileId {
        self.frange.file_id.file_id(self.db())
    }

    pub(crate) fn file_id(&self) -> EditionedFileId {
        self.frange.file_id
    }

    pub(crate) fn edition(&self) -> Edition {
        self.frange.file_id.edition(self.db())
    }

    pub(crate) fn has_empty_selection(&self) -> bool {
        self.trimmed_range.is_empty()
    }

    /// Returns the selected range trimmed for whitespace tokens, that is the range will be snapped
    /// to the nearest enclosed token.
    pub(crate) fn selection_trimmed(&self) -> TextRange {
        self.trimmed_range
    }

    pub(crate) fn source_file(&self) -> &SourceFile {
        &self.source_file
    }

    pub(crate) fn token_at_offset(&self) -> TokenAtOffset<SyntaxToken> {
        self.token_at_offset.clone()
    }
    pub(crate) fn find_token_syntax_at_offset(&self, kind: SyntaxKind) -> Option<SyntaxToken> {
        self.token_at_offset().find(|it| it.kind() == kind)
    }
    pub(crate) fn find_token_at_offset<T: AstToken>(&self) -> Option<T> {
        self.token_at_offset().find_map(T::cast)
    }
    pub(crate) fn find_node_at_offset<N: AstNode>(&self) -> Option<N> {
        find_node_at_offset(self.source_file.syntax(), self.offset())
    }
    pub(crate) fn find_node_at_trimmed_offset<N: AstNode>(&self) -> Option<N> {
        find_node_at_offset(self.source_file.syntax(), self.trimmed_range.start())
    }
    pub(crate) fn find_node_at_range<N: AstNode>(&self) -> Option<N> {
        find_node_at_range(self.source_file.syntax(), self.trimmed_range)
    }
    pub(crate) fn find_node_at_offset_with_descend<N: AstNode>(&self) -> Option<N> {
        self.sema.find_node_at_offset_with_descend(self.source_file.syntax(), self.offset())
    }
    /// Returns the element covered by the selection range, this excludes trailing whitespace in the selection.
    pub(crate) fn covering_element(&self) -> SyntaxElement {
        self.covering_element.clone()
    }
}

pub(crate) struct Assists {
    file: FileId,
    resolve: AssistResolveStrategy,
    buf: Vec<Assist>,
    allowed: Option<Vec<AssistKind>>,
}

impl Assists {
    pub(crate) fn new(ctx: &AssistContext<'_>, resolve: AssistResolveStrategy) -> Assists {
        Assists {
            resolve,
            file: ctx.frange.file_id.file_id(ctx.db()),
            buf: Vec::new(),
            allowed: ctx.config.allowed.clone(),
        }
    }

    pub(crate) fn finish(mut self) -> Vec<Assist> {
        self.buf.sort_by_key(|assist| assist.target.len());
        self.buf
    }

    pub(crate) fn add(
        &mut self,
        id: AssistId,
        label: impl Into<String>,
        target: TextRange,
        f: impl FnOnce(&mut SourceChangeBuilder),
    ) -> Option<()> {
        let mut f = Some(f);
        self.add_impl(None, id, label.into(), target, &mut |it| f.take().unwrap()(it))
    }

    pub(crate) fn add_group(
        &mut self,
        group: &GroupLabel,
        id: AssistId,
        label: impl Into<String>,
        target: TextRange,
        f: impl FnOnce(&mut SourceChangeBuilder),
    ) -> Option<()> {
        let mut f = Some(f);
        self.add_impl(Some(group), id, label.into(), target, &mut |it| f.take().unwrap()(it))
    }

    fn add_impl(
        &mut self,
        group: Option<&GroupLabel>,
        id: AssistId,
        label: String,
        target: TextRange,
        f: &mut dyn FnMut(&mut SourceChangeBuilder),
    ) -> Option<()> {
        if !self.is_allowed(&id) {
            return None;
        }

        let mut command = None;
        let source_change = if self.resolve.should_resolve(&id) {
            let mut builder = SourceChangeBuilder::new(self.file);
            f(&mut builder);
            command = builder.command.take();
            Some(builder.finish())
        } else {
            None
        };

        let label = Label::new(label);
        let group = group.cloned();
        self.buf.push(Assist { id, label, group, target, source_change, command });
        Some(())
    }

    fn is_allowed(&self, id: &AssistId) -> bool {
        match &self.allowed {
            Some(allowed) => allowed.iter().any(|kind| kind.contains(id.1)),
            None => true,
        }
    }
}
