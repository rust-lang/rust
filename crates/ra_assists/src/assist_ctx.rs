//! This module defines `AssistCtx` -- the API surface that is exposed to assists.
use hir::Semantics;
use ra_db::FileRange;
use ra_fmt::{leading_indent, reindent};
use ra_ide_db::RootDatabase;
use ra_syntax::{
    algo::{self, find_covering_element, find_node_at_offset},
    AstNode, SourceFile, SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken, TextRange, TextUnit,
    TokenAtOffset,
};
use ra_text_edit::TextEditBuilder;

use crate::{AssistAction, AssistId, AssistLabel, GroupLabel, ResolvedAssist};
use algo::SyntaxRewriter;

#[derive(Clone, Debug)]
pub(crate) struct Assist(pub(crate) Vec<AssistInfo>);

#[derive(Clone, Debug)]
pub(crate) struct AssistInfo {
    pub(crate) label: AssistLabel,
    pub(crate) group_label: Option<GroupLabel>,
    pub(crate) action: Option<AssistAction>,
}

impl AssistInfo {
    fn new(label: AssistLabel) -> AssistInfo {
        AssistInfo { label, group_label: None, action: None }
    }

    fn resolved(self, action: AssistAction) -> AssistInfo {
        AssistInfo { action: Some(action), ..self }
    }

    fn with_group(self, group_label: GroupLabel) -> AssistInfo {
        AssistInfo { group_label: Some(group_label), ..self }
    }

    pub(crate) fn into_resolved(self) -> Option<ResolvedAssist> {
        let label = self.label;
        let group_label = self.group_label;
        self.action.map(|action| ResolvedAssist { label, group_label, action })
    }
}

pub(crate) type AssistHandler = fn(AssistCtx) -> Option<Assist>;

/// `AssistCtx` allows to apply an assist or check if it could be applied.
///
/// Assists use a somewhat over-engineered approach, given the current needs. The
/// assists workflow consists of two phases. In the first phase, a user asks for
/// the list of available assists. In the second phase, the user picks a
/// particular assist and it gets applied.
///
/// There are two peculiarities here:
///
/// * first, we ideally avoid computing more things then necessary to answer
///   "is assist applicable" in the first phase.
/// * second, when we are applying assist, we don't have a guarantee that there
///   weren't any changes between the point when user asked for assists and when
///   they applied a particular assist. So, when applying assist, we need to do
///   all the checks from scratch.
///
/// To avoid repeating the same code twice for both "check" and "apply"
/// functions, we use an approach reminiscent of that of Django's function based
/// views dealing with forms. Each assist receives a runtime parameter,
/// `should_compute_edit`. It first check if an edit is applicable (potentially
/// computing info required to compute the actual edit). If it is applicable,
/// and `should_compute_edit` is `true`, it then computes the actual edit.
///
/// So, to implement the original assists workflow, we can first apply each edit
/// with `should_compute_edit = false`, and then applying the selected edit
/// again, with `should_compute_edit = true` this time.
///
/// Note, however, that we don't actually use such two-phase logic at the
/// moment, because the LSP API is pretty awkward in this place, and it's much
/// easier to just compute the edit eagerly :-)
#[derive(Clone)]
pub(crate) struct AssistCtx<'a> {
    pub(crate) sema: &'a Semantics<'a, RootDatabase>,
    pub(crate) db: &'a RootDatabase,
    pub(crate) frange: FileRange,
    source_file: SourceFile,
    should_compute_edit: bool,
}

impl<'a> AssistCtx<'a> {
    pub fn new(
        sema: &'a Semantics<'a, RootDatabase>,
        frange: FileRange,
        should_compute_edit: bool,
    ) -> AssistCtx<'a> {
        let source_file = sema.parse(frange.file_id);
        AssistCtx { sema, db: sema.db, frange, source_file, should_compute_edit }
    }

    pub(crate) fn add_assist(
        self,
        id: AssistId,
        label: impl Into<String>,
        f: impl FnOnce(&mut ActionBuilder),
    ) -> Option<Assist> {
        let label = AssistLabel::new(label.into(), id);

        let mut info = AssistInfo::new(label);
        if self.should_compute_edit {
            let action = {
                let mut edit = ActionBuilder::default();
                f(&mut edit);
                edit.build()
            };
            info = info.resolved(action)
        };

        Some(Assist(vec![info]))
    }

    pub(crate) fn add_assist_group(self, group_name: impl Into<String>) -> AssistGroup<'a> {
        AssistGroup { ctx: self, group_name: group_name.into(), assists: Vec::new() }
    }

    pub(crate) fn token_at_offset(&self) -> TokenAtOffset<SyntaxToken> {
        self.source_file.syntax().token_at_offset(self.frange.range.start())
    }

    pub(crate) fn find_token_at_offset(&self, kind: SyntaxKind) -> Option<SyntaxToken> {
        self.token_at_offset().find(|it| it.kind() == kind)
    }

    pub(crate) fn find_node_at_offset<N: AstNode>(&self) -> Option<N> {
        find_node_at_offset(self.source_file.syntax(), self.frange.range.start())
    }
    pub(crate) fn covering_element(&self) -> SyntaxElement {
        find_covering_element(self.source_file.syntax(), self.frange.range)
    }
    pub(crate) fn covering_node_for_range(&self, range: TextRange) -> SyntaxElement {
        find_covering_element(self.source_file.syntax(), range)
    }
}

pub(crate) struct AssistGroup<'a> {
    ctx: AssistCtx<'a>,
    group_name: String,
    assists: Vec<AssistInfo>,
}

impl<'a> AssistGroup<'a> {
    pub(crate) fn add_assist(
        &mut self,
        id: AssistId,
        label: impl Into<String>,
        f: impl FnOnce(&mut ActionBuilder),
    ) {
        let label = AssistLabel::new(label.into(), id);

        let mut info = AssistInfo::new(label).with_group(GroupLabel(self.group_name.clone()));
        if self.ctx.should_compute_edit {
            let action = {
                let mut edit = ActionBuilder::default();
                f(&mut edit);
                edit.build()
            };
            info = info.resolved(action)
        };

        self.assists.push(info)
    }

    pub(crate) fn finish(self) -> Option<Assist> {
        if self.assists.is_empty() {
            None
        } else {
            Some(Assist(self.assists))
        }
    }
}

#[derive(Default)]
pub(crate) struct ActionBuilder {
    edit: TextEditBuilder,
    cursor_position: Option<TextUnit>,
    target: Option<TextRange>,
}

impl ActionBuilder {
    /// Replaces specified `range` of text with a given string.
    pub(crate) fn replace(&mut self, range: TextRange, replace_with: impl Into<String>) {
        self.edit.replace(range, replace_with.into())
    }

    /// Replaces specified `node` of text with a given string, reindenting the
    /// string to maintain `node`'s existing indent.
    // FIXME: remove in favor of ra_syntax::edit::IndentLevel::increase_indent
    pub(crate) fn replace_node_and_indent(
        &mut self,
        node: &SyntaxNode,
        replace_with: impl Into<String>,
    ) {
        let mut replace_with = replace_with.into();
        if let Some(indent) = leading_indent(node) {
            replace_with = reindent(&replace_with, &indent)
        }
        self.replace(node.text_range(), replace_with)
    }

    /// Remove specified `range` of text.
    #[allow(unused)]
    pub(crate) fn delete(&mut self, range: TextRange) {
        self.edit.delete(range)
    }

    /// Append specified `text` at the given `offset`
    pub(crate) fn insert(&mut self, offset: TextUnit, text: impl Into<String>) {
        self.edit.insert(offset, text.into())
    }

    /// Specify desired position of the cursor after the assist is applied.
    pub(crate) fn set_cursor(&mut self, offset: TextUnit) {
        self.cursor_position = Some(offset)
    }

    /// Specify that the assist should be active withing the `target` range.
    ///
    /// Target ranges are used to sort assists: the smaller the target range,
    /// the more specific assist is, and so it should be sorted first.
    pub(crate) fn target(&mut self, target: TextRange) {
        self.target = Some(target)
    }

    /// Get access to the raw `TextEditBuilder`.
    pub(crate) fn text_edit_builder(&mut self) -> &mut TextEditBuilder {
        &mut self.edit
    }

    pub(crate) fn replace_ast<N: AstNode>(&mut self, old: N, new: N) {
        algo::diff(old.syntax(), new.syntax()).into_text_edit(&mut self.edit)
    }
    pub(crate) fn rewrite(&mut self, rewriter: SyntaxRewriter) {
        let node = rewriter.rewrite_root().unwrap();
        let new = rewriter.rewrite(&node);
        algo::diff(&node, &new).into_text_edit(&mut self.edit)
    }

    fn build(self) -> AssistAction {
        AssistAction {
            edit: self.edit.finish(),
            cursor_position: self.cursor_position,
            target: self.target,
        }
    }
}
