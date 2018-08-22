use {TextUnit, EditBuilder, Edit};
use libsyntax2::{
    ast::{self, AstNode, AttrsOwner, TypeParamsOwner, NameOwner, ParsedFile},
    SyntaxKind::COMMA,
    SyntaxNodeRef,
    algo::{
        Direction, siblings,
        find_leaf_at_offset, ancestors,
    },
};

pub struct ActionResult {
    pub edit: Edit,
    pub cursor_position: Option<TextUnit>,
}

pub fn flip_comma<'a>(file: &'a ParsedFile, offset: TextUnit) -> Option<impl FnOnce() -> ActionResult + 'a> {
    let syntax = file.syntax();

    let comma = find_leaf_at_offset(syntax, offset).find(|leaf| leaf.kind() == COMMA)?;
    let left = non_trivia_sibling(comma, Direction::Backward)?;
    let right = non_trivia_sibling(comma, Direction::Forward)?;
    Some(move || {
        let mut edit = EditBuilder::new();
        edit.replace(left.range(), right.text());
        edit.replace(right.range(), left.text());
        ActionResult {
            edit: edit.finish(),
            cursor_position: None,
        }
    })
}

pub fn add_derive<'a>(file: &'a ParsedFile, offset: TextUnit) -> Option<impl FnOnce() -> ActionResult + 'a> {
    let nominal = find_node::<ast::NominalDef>(file.syntax(), offset)?;
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
            cursor_position: Some(offset),
        }
    })
}

pub fn add_impl<'a>(file: &'a ParsedFile, offset: TextUnit) -> Option<impl FnOnce() -> ActionResult + 'a> {
    let nominal = find_node::<ast::NominalDef>(file.syntax(), offset)?;
    let name = nominal.name()?;

    Some(move || {
        // let type_params = nominal.type_param_list();
        // let type_args = match type_params {
        //     None => String::new(),
        //     Some(params) => {
        //         let mut buf = String::new();
        //     }
        // };
        let mut edit = EditBuilder::new();
        let start_offset = nominal.syntax().range().end();
        edit.insert(
            start_offset,
            format!(
                "\n\nimpl {} {{\n\n}}",
                name.text(),
            )
        );
        ActionResult {
            edit: edit.finish(),
            cursor_position: Some(
                start_offset + TextUnit::of_str("\n\nimpl  {\n") + name.syntax().range().len()
            ),
        }
    })
}

fn non_trivia_sibling(node: SyntaxNodeRef, direction: Direction) -> Option<SyntaxNodeRef> {
    siblings(node, direction)
        .skip(1)
        .find(|node| !node.kind().is_trivia())
}

pub fn find_node<'a, N: AstNode<'a>>(syntax: SyntaxNodeRef<'a>, offset: TextUnit) -> Option<N> {
    let leaves = find_leaf_at_offset(syntax, offset);
    let leaf = leaves.clone()
        .find(|leaf| !leaf.kind().is_trivia())
        .or_else(|| leaves.right_biased())?;
    ancestors(leaf)
        .filter_map(N::cast)
        .next()
}

