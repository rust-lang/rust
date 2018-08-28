use join_to_string::join;

use libsyntax2::{
    File, TextUnit,
    ast::{self, AstNode, AttrsOwner, TypeParamsOwner, NameOwner},
    SyntaxKind::COMMA,
    SyntaxNodeRef,
    algo::{
        Direction, siblings,
        find_leaf_at_offset,
    },
};

use {EditBuilder, Edit, find_node_at_offset};

#[derive(Debug)]
pub struct ActionResult {
    pub edit: Edit,
    pub cursor_position: Option<TextUnit>,
}

pub fn flip_comma<'a>(file: &'a File, offset: TextUnit) -> Option<impl FnOnce() -> ActionResult + 'a> {
    let syntax = file.syntax();

    let comma = find_leaf_at_offset(syntax, offset).find(|leaf| leaf.kind() == COMMA)?;
    let left = non_trivia_sibling(comma, Direction::Backward)?;
    let right = non_trivia_sibling(comma, Direction::Forward)?;
    Some(move || {
        let mut edit = EditBuilder::new();
        edit.replace(left.range(), right.text().to_string());
        edit.replace(right.range(), left.text().to_string());
        ActionResult {
            edit: edit.finish(),
            cursor_position: None,
        }
    })
}

pub fn add_derive<'a>(file: &'a File, offset: TextUnit) -> Option<impl FnOnce() -> ActionResult + 'a> {
    let nominal = find_node_at_offset::<ast::NominalDef>(file.syntax(), offset)?;
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

pub fn add_impl<'a>(file: &'a File, offset: TextUnit) -> Option<impl FnOnce() -> ActionResult + 'a> {
    let nominal = find_node_at_offset::<ast::NominalDef>(file.syntax(), offset)?;
    let name = nominal.name()?;

    Some(move || {
        let type_params = nominal.type_param_list();
        let mut edit = EditBuilder::new();
        let start_offset = nominal.syntax().range().end();
        let mut buf = String::new();
        buf.push_str("\n\nimpl");
        if let Some(type_params) = type_params {
            type_params.syntax().text()
                .push_to(&mut buf);
        }
        buf.push_str(" ");
        buf.push_str(name.text().as_str());
        if let Some(type_params) = type_params {
            let lifetime_params = type_params.lifetime_params().filter_map(|it| it.lifetime()).map(|it| it.text());
            let type_params = type_params.type_params().filter_map(|it| it.name()).map(|it| it.text());
            join(lifetime_params.chain(type_params))
                .surround_with("<", ">")
                .to_buf(&mut buf);
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

#[cfg(test)]
mod tests {
    use super::*;
    use test_utils::check_action;

    #[test]
    fn test_swap_comma() {
        check_action(
            "fn foo(x: i32,<|> y: Result<(), ()>) {}",
            "fn foo(y: Result<(), ()>,<|> x: i32) {}",
            |file, off| flip_comma(file, off).map(|f| f()),
        )
    }

    #[test]
    fn test_add_derive() {
        check_action(
            "struct Foo { a: i32, <|>}",
            "#[derive(<|>)]\nstruct Foo { a: i32, }",
            |file, off| add_derive(file, off).map(|f| f()),
        );
        check_action(
            "struct Foo { <|> a: i32, }",
            "#[derive(<|>)]\nstruct Foo {  a: i32, }",
            |file, off| add_derive(file, off).map(|f| f()),
        );
        check_action(
            "#[derive(Clone)]\nstruct Foo { a: i32<|>, }",
            "#[derive(Clone<|>)]\nstruct Foo { a: i32, }",
            |file, off| add_derive(file, off).map(|f| f()),
        );
    }

    #[test]
    fn test_add_impl() {
        check_action(
            "struct Foo {<|>}\n",
            "struct Foo {}\n\nimpl Foo {\n<|>\n}\n",
            |file, off| add_impl(file, off).map(|f| f()),
        );
        check_action(
            "struct Foo<T: Clone> {<|>}",
            "struct Foo<T: Clone> {}\n\nimpl<T: Clone> Foo<T> {\n<|>\n}",
            |file, off| add_impl(file, off).map(|f| f()),
        );
        check_action(
            "struct Foo<'a, T: Foo<'a>> {<|>}",
            "struct Foo<'a, T: Foo<'a>> {}\n\nimpl<'a, T: Foo<'a>> Foo<'a, T> {\n<|>\n}",
            |file, off| add_impl(file, off).map(|f| f()),
        );
    }

}
