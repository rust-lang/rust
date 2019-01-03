use ra_text_edit::TextEditBuilder;
use ra_syntax::{
    algo::find_leaf_at_offset,
    Direction, SourceFileNode,
    SyntaxKind::COMMA,
    TextUnit,
};

use crate::assists::{LocalEdit, non_trivia_sibling};

pub fn flip_comma<'a>(
    file: &'a SourceFileNode,
    offset: TextUnit,
) -> Option<impl FnOnce() -> LocalEdit + 'a> {
    let syntax = file.syntax();

    let comma = find_leaf_at_offset(syntax, offset).find(|leaf| leaf.kind() == COMMA)?;
    let prev = non_trivia_sibling(comma, Direction::Prev)?;
    let next = non_trivia_sibling(comma, Direction::Next)?;
    Some(move || {
        let mut edit = TextEditBuilder::new();
        edit.replace(prev.range(), next.text().to_string());
        edit.replace(next.range(), prev.text().to_string());
        LocalEdit {
            label: "flip comma".to_string(),
            edit: edit.finish(),
            cursor_position: None,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::check_action;

    #[test]
    fn test_swap_comma() {
        check_action(
            "fn foo(x: i32,<|> y: Result<(), ()>) {}",
            "fn foo(y: Result<(), ()>,<|> x: i32) {}",
            |file, off| flip_comma(file, off).map(|f| f()),
        )
    }
}
