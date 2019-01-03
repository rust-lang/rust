//! This modules contains various "assits": suggestions for source code edits
//! which are likely to occur at a given cursor positon. For example, if the
//! cursor is on the `,`, a possible assist is swapping the elments around the
//! comma.

mod flip_comma;
mod add_derive;
mod add_impl;
mod introduce_variable;
mod change_visibility;

use ra_text_edit::TextEdit;
use ra_syntax::{Direction, SyntaxNodeRef, TextUnit};

pub use self::{
    flip_comma::flip_comma,
    add_derive::add_derive,
    add_impl::add_impl,
    introduce_variable::introduce_variable,
    change_visibility::change_visibility,
};

#[derive(Debug)]
pub struct LocalEdit {
    pub label: String,
    pub edit: TextEdit,
    pub cursor_position: Option<TextUnit>,
}

fn non_trivia_sibling(node: SyntaxNodeRef, direction: Direction) -> Option<SyntaxNodeRef> {
    node.siblings(direction)
        .skip(1)
        .find(|node| !node.kind().is_trivia())
}
