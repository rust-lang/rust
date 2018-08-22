use std::{
    fmt::{self, Write},
};

use libsyntax2::{
    ast::{self, AstNode, AttrsOwner, TypeParamsOwner, NameOwner, ParsedFile},
    SyntaxKind::COMMA,
    SyntaxNodeRef,
    algo::{
        Direction, siblings,
        find_leaf_at_offset, ancestors,
    },
};

use {TextUnit, EditBuilder, Edit};

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
        let type_params = nominal.type_param_list();
        let mut edit = EditBuilder::new();
        let start_offset = nominal.syntax().range().end();
        let mut buf = String::new();
        buf.push_str("\n\nimpl");
        if let Some(type_params) = type_params {
            buf.push_display(&type_params.syntax().text());
        }
        buf.push_str(" ");
        buf.push_str(name.text().as_str());
        if let Some(type_params) = type_params {
            comma_list(
                &mut buf, "<", ">",
                type_params.type_params()
                    .filter_map(|it| it.name())
                    .map(|it| it.text())
            );
        }
        buf.push_str(" {\n");
        let offset = start_offset + TextUnit::of_str(&buf);
        buf.push_str("\n}");
        edit.insert(start_offset, buf);
        ActionResult {
            edit: edit.finish(),
            cursor_position: Some(offset),
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

fn comma_list(buf: &mut String, bra: &str, ket: &str, items: impl Iterator<Item=impl fmt::Display>) {
    buf.push_str(bra);
    let mut first = true;
    for item in items {
        if !first {
            first = false;
            buf.push_str(", ");
        }
        write!(buf, "{}", item).unwrap();
    }
    buf.push_str(ket);
}

trait PushDisplay {
    fn push_display<T: fmt::Display>(&mut self, item: &T);
}

impl PushDisplay for String {
    fn push_display<T: fmt::Display>(&mut self, item: &T) {
        use std::fmt::Write;
        write!(self, "{}", item).unwrap()
    }
}
