use {TextUnit, File, EditBuilder, Edit};
use libsyntax2::{
    ast::AstNode,
    SyntaxKind::COMMA,
    SyntaxNodeRef,
    algo::{
        Direction, siblings,
        find_leaf_at_offset,
    },
};

pub fn flip_comma<'a>(file: &'a File, offset: TextUnit) -> Option<impl FnOnce() -> Edit + 'a> {
    let syntax = file.syntax();
    let syntax = syntax.as_ref();

    let comma = find_leaf_at_offset(syntax, offset).find(|leaf| leaf.kind() == COMMA)?;
    let left = non_trivia_sibling(comma, Direction::Backward)?;
    let right = non_trivia_sibling(comma, Direction::Forward)?;
    Some(move || {
        let mut edit = EditBuilder::new();
        edit.replace(left.range(), right.text());
        edit.replace(right.range(), left.text());
        edit.finish()
    })
}

fn non_trivia_sibling(node: SyntaxNodeRef, direction: Direction) -> Option<SyntaxNodeRef> {
    siblings(node, direction)
        .skip(1)
        .find(|node| !node.kind().is_trivia())
}


