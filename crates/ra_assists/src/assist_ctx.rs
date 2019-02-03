use hir::db::HirDatabase;
use ra_text_edit::TextEditBuilder;
use ra_db::FileRange;
use ra_syntax::{
    SourceFile, TextRange, AstNode, TextUnit, SyntaxNode,
    algo::{find_leaf_at_offset, find_node_at_offset, find_covering_node, LeafAtOffset},
};
use ra_ide_api_light::formatting::{leading_indent, reindent};

use crate::{AssistLabel, AssistAction};

pub(crate) enum Assist {
    Unresolved(AssistLabel),
    Resolved(AssistLabel, AssistAction),
}

/// `AssistCtx` allows to apply an assist or check if it could be applied.
///
/// Assists use a somewhat overengineered approach, given the current needs. The
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
/// easier to just compute the edit eagerly :-)#[derive(Debug, Clone)]
#[derive(Debug)]
pub(crate) struct AssistCtx<'a, DB> {
    pub(crate) db: &'a DB,
    pub(crate) frange: FileRange,
    source_file: &'a SourceFile,
    should_compute_edit: bool,
}

impl<'a, DB> Clone for AssistCtx<'a, DB> {
    fn clone(&self) -> Self {
        AssistCtx {
            db: self.db,
            frange: self.frange,
            source_file: self.source_file,
            should_compute_edit: self.should_compute_edit,
        }
    }
}

impl<'a, DB: HirDatabase> AssistCtx<'a, DB> {
    pub(crate) fn with_ctx<F, T>(db: &DB, frange: FileRange, should_compute_edit: bool, f: F) -> T
    where
        F: FnOnce(AssistCtx<DB>) -> T,
    {
        let source_file = &db.parse(frange.file_id);
        let ctx = AssistCtx {
            db,
            frange,
            source_file,
            should_compute_edit,
        };
        f(ctx)
    }

    pub(crate) fn build(
        self,
        label: impl Into<String>,
        f: impl FnOnce(&mut AssistBuilder),
    ) -> Option<Assist> {
        let label = AssistLabel {
            label: label.into(),
        };
        if !self.should_compute_edit {
            return Some(Assist::Unresolved(label));
        }
        let action = {
            let mut edit = AssistBuilder::default();
            f(&mut edit);
            edit.build()
        };
        Some(Assist::Resolved(label, action))
    }

    pub(crate) fn leaf_at_offset(&self) -> LeafAtOffset<&'a SyntaxNode> {
        find_leaf_at_offset(self.source_file.syntax(), self.frange.range.start())
    }

    pub(crate) fn node_at_offset<N: AstNode>(&self) -> Option<&'a N> {
        find_node_at_offset(self.source_file.syntax(), self.frange.range.start())
    }
    pub(crate) fn covering_node(&self) -> &'a SyntaxNode {
        find_covering_node(self.source_file.syntax(), self.frange.range)
    }
}

#[derive(Default)]
pub(crate) struct AssistBuilder {
    edit: TextEditBuilder,
    cursor_position: Option<TextUnit>,
}

impl AssistBuilder {
    pub(crate) fn replace(&mut self, range: TextRange, replace_with: impl Into<String>) {
        self.edit.replace(range, replace_with.into())
    }

    pub(crate) fn replace_node_and_indent(
        &mut self,
        node: &SyntaxNode,
        replace_with: impl Into<String>,
    ) {
        let mut replace_with = replace_with.into();
        if let Some(indent) = leading_indent(node) {
            replace_with = reindent(&replace_with, indent)
        }
        self.replace(node.range(), replace_with)
    }

    #[allow(unused)]
    pub(crate) fn delete(&mut self, range: TextRange) {
        self.edit.delete(range)
    }

    pub(crate) fn insert(&mut self, offset: TextUnit, text: impl Into<String>) {
        self.edit.insert(offset, text.into())
    }

    pub(crate) fn set_cursor(&mut self, offset: TextUnit) {
        self.cursor_position = Some(offset)
    }

    fn build(self) -> AssistAction {
        AssistAction {
            edit: self.edit.finish(),
            cursor_position: self.cursor_position,
        }
    }
}
