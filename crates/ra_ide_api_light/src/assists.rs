//! This modules contains various "assits": suggestions for source code edits
//! which are likely to occur at a given cursor positon. For example, if the
//! cursor is on the `,`, a possible assist is swapping the elments around the
//! comma.

mod flip_comma;
mod add_derive;
mod add_impl;
mod introduce_variable;
mod change_visibility;
mod split_import;
mod replace_if_let_with_match;

use ra_text_edit::{TextEdit, TextEditBuilder};
use ra_syntax::{
    Direction, SyntaxNode, TextUnit, TextRange, SourceFile, AstNode,
    algo::{find_leaf_at_offset, find_node_at_offset, find_covering_node, LeafAtOffset},
};
use itertools::Itertools;

use crate::formatting::leading_indent;

pub use self::{
    flip_comma::flip_comma,
    add_derive::add_derive,
    add_impl::add_impl,
    introduce_variable::introduce_variable,
    change_visibility::change_visibility,
    split_import::split_import,
    replace_if_let_with_match::replace_if_let_with_match,
};

/// Return all the assists applicable at the given position.
pub fn assists(file: &SourceFile, range: TextRange) -> Vec<LocalEdit> {
    let ctx = AssistCtx::new(file, range);
    [
        flip_comma,
        add_derive,
        add_impl,
        introduce_variable,
        change_visibility,
        split_import,
        replace_if_let_with_match,
    ]
    .iter()
    .filter_map(|&assist| ctx.clone().apply(assist))
    .collect()
}

#[derive(Debug)]
pub struct LocalEdit {
    pub label: String,
    pub edit: TextEdit,
    pub cursor_position: Option<TextUnit>,
}

fn non_trivia_sibling(node: &SyntaxNode, direction: Direction) -> Option<&SyntaxNode> {
    node.siblings(direction)
        .skip(1)
        .find(|node| !node.kind().is_trivia())
}

/// `AssistCtx` allows to apply an assist or check if it could be applied.
///
/// Assists use a somewhat overengeneered approach, given the current needs. The
/// assists workflow consists of two phases. In the first phase, a user asks for
/// the list of available assists. In the second phase, the user picks a
/// particular assist and it gets applied.
///
/// There are two peculiarities here:
///
/// * first, we ideally avoid computing more things then neccessary to answer
///   "is assist applicable" in the first phase.
/// * second, when we are appling assist, we don't have a gurantee that there
///   weren't any changes between the point when user asked for assists and when
///   they applied a particular assist. So, when applying assist, we need to do
///   all the checks from scratch.
///
/// To avoid repeating the same code twice for both "check" and "apply"
/// functions, we use an approach remeniscent of that of Django's function based
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
/// easier to just compute the edit eagarly :-)
#[derive(Debug, Clone)]
pub struct AssistCtx<'a> {
    source_file: &'a SourceFile,
    range: TextRange,
    should_compute_edit: bool,
}

#[derive(Debug)]
pub enum Assist {
    Applicable,
    Edit(LocalEdit),
}

#[derive(Default)]
struct AssistBuilder {
    edit: TextEditBuilder,
    cursor_position: Option<TextUnit>,
}

impl<'a> AssistCtx<'a> {
    pub fn new(source_file: &'a SourceFile, range: TextRange) -> AssistCtx {
        AssistCtx {
            source_file,
            range,
            should_compute_edit: false,
        }
    }

    pub fn apply(mut self, assist: fn(AssistCtx) -> Option<Assist>) -> Option<LocalEdit> {
        self.should_compute_edit = true;
        match assist(self) {
            None => None,
            Some(Assist::Edit(e)) => Some(e),
            Some(Assist::Applicable) => unreachable!(),
        }
    }

    pub fn check(mut self, assist: fn(AssistCtx) -> Option<Assist>) -> bool {
        self.should_compute_edit = false;
        match assist(self) {
            None => false,
            Some(Assist::Edit(_)) => unreachable!(),
            Some(Assist::Applicable) => true,
        }
    }

    fn build(self, label: impl Into<String>, f: impl FnOnce(&mut AssistBuilder)) -> Option<Assist> {
        if !self.should_compute_edit {
            return Some(Assist::Applicable);
        }
        let mut edit = AssistBuilder::default();
        f(&mut edit);
        Some(Assist::Edit(LocalEdit {
            label: label.into(),
            edit: edit.edit.finish(),
            cursor_position: edit.cursor_position,
        }))
    }

    pub(crate) fn leaf_at_offset(&self) -> LeafAtOffset<&'a SyntaxNode> {
        find_leaf_at_offset(self.source_file.syntax(), self.range.start())
    }
    pub(crate) fn node_at_offset<N: AstNode>(&self) -> Option<&'a N> {
        find_node_at_offset(self.source_file.syntax(), self.range.start())
    }
    pub(crate) fn covering_node(&self) -> &'a SyntaxNode {
        find_covering_node(self.source_file.syntax(), self.range)
    }
}

impl AssistBuilder {
    fn replace(&mut self, range: TextRange, replace_with: impl Into<String>) {
        self.edit.replace(range, replace_with.into())
    }
    fn replace_node_and_indent(&mut self, node: &SyntaxNode, replace_with: impl Into<String>) {
        let mut replace_with = replace_with.into();
        if let Some(indent) = leading_indent(node) {
            replace_with = reindent(&replace_with, indent)
        }
        self.replace(node.range(), replace_with)
    }
    #[allow(unused)]
    fn delete(&mut self, range: TextRange) {
        self.edit.delete(range)
    }
    fn insert(&mut self, offset: TextUnit, text: impl Into<String>) {
        self.edit.insert(offset, text.into())
    }
    fn set_cursor(&mut self, offset: TextUnit) {
        self.cursor_position = Some(offset)
    }
}

fn reindent(text: &str, indent: &str) -> String {
    let indent = format!("\n{}", indent);
    text.lines().intersperse(&indent).collect()
}

#[cfg(test)]
fn check_assist(assist: fn(AssistCtx) -> Option<Assist>, before: &str, after: &str) {
    crate::test_utils::check_action(before, after, |file, off| {
        let range = TextRange::offset_len(off, 0.into());
        AssistCtx::new(file, range).apply(assist)
    })
}

#[cfg(test)]
fn check_assist_range(assist: fn(AssistCtx) -> Option<Assist>, before: &str, after: &str) {
    crate::test_utils::check_action_range(before, after, |file, range| {
        AssistCtx::new(file, range).apply(assist)
    })
}
