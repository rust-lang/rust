use {TextUnit, File, EditBuilder, Edit};
use libsyntax2::{
    ast::{self, AstNode, AttrsOwner},
    SyntaxKind::COMMA,
    SyntaxNodeRef,
    SyntaxRoot,
    algo::{
        Direction, siblings,
        find_leaf_at_offset, ancestors,
    },
};

pub struct ActionResult {
    pub edit: Edit,
    pub cursor_position: CursorPosition,
}

pub enum CursorPosition {
    Same,
    Offset(TextUnit),
}

pub fn flip_comma<'a>(file: &'a File, offset: TextUnit) -> Option<impl FnOnce() -> ActionResult + 'a> {
    let syntax = file.syntax();
    let syntax = syntax.as_ref();

    let comma = find_leaf_at_offset(syntax, offset).find(|leaf| leaf.kind() == COMMA)?;
    let left = non_trivia_sibling(comma, Direction::Backward)?;
    let right = non_trivia_sibling(comma, Direction::Forward)?;
    Some(move || {
        let mut edit = EditBuilder::new();
        edit.replace(left.range(), right.text());
        edit.replace(right.range(), left.text());
        ActionResult {
            edit: edit.finish(),
            cursor_position: CursorPosition::Same,
        }
    })
}

pub fn add_derive<'a>(file: &'a File, offset: TextUnit) -> Option<impl FnOnce() -> ActionResult + 'a> {
    let nominal = find_node::<ast::NominalDef<_>>(file.syntax_ref(), offset)?;
    Some(move || {
        let derive_attr = nominal
            .attrs()
            .filter_map(|x| x.as_call())
            .filter(|(name, _arg)| name == "derive")
            .map(|(_name, arg)| arg)
            .next();
        let mut edit = EditBuilder::new();
        let offset = match derive_attr {
            None => {
                let node_start = nominal.syntax().range().start();
                edit.insert(node_start, "#[derive()]\n".to_string());
                node_start + TextUnit::of_str("#[derive(")
            }
            Some(tt) => {
                tt.syntax().range().end() - TextUnit::of_char(')')
            }
        };
        ActionResult {
            edit: edit.finish(),
            cursor_position: CursorPosition::Offset(offset),
        }
    })
}

fn non_trivia_sibling(node: SyntaxNodeRef, direction: Direction) -> Option<SyntaxNodeRef> {
    siblings(node, direction)
        .skip(1)
        .find(|node| !node.kind().is_trivia())
}

pub fn find_node<'a, N: AstNode<&'a SyntaxRoot>>(syntax: SyntaxNodeRef<'a>, offset: TextUnit) -> Option<N> {
    let leaves = find_leaf_at_offset(syntax, offset);
    let leaf = leaves.clone()
        .find(|leaf| !leaf.kind().is_trivia())
        .or_else(|| leaves.right_biased())?;
    ancestors(leaf)
        .filter_map(N::cast)
        .next()
}

