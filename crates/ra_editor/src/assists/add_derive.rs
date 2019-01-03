use ra_text_edit::TextEditBuilder;
use ra_syntax::{
    ast::{self, AstNode, AttrsOwner},
    SourceFileNode,
    SyntaxKind::{WHITESPACE, COMMENT},
    TextUnit,
};

use crate::{
    find_node_at_offset,
    assists::LocalEdit,
};

pub fn add_derive<'a>(
    file: &'a SourceFileNode,
    offset: TextUnit,
) -> Option<impl FnOnce() -> LocalEdit + 'a> {
    let nominal = find_node_at_offset::<ast::NominalDef>(file.syntax(), offset)?;
    let node_start = derive_insertion_offset(nominal)?;
    return Some(move || {
        let derive_attr = nominal
            .attrs()
            .filter_map(|x| x.as_call())
            .filter(|(name, _arg)| name == "derive")
            .map(|(_name, arg)| arg)
            .next();
        let mut edit = TextEditBuilder::new();
        let offset = match derive_attr {
            None => {
                edit.insert(node_start, "#[derive()]\n".to_string());
                node_start + TextUnit::of_str("#[derive(")
            }
            Some(tt) => tt.syntax().range().end() - TextUnit::of_char(')'),
        };
        LocalEdit {
            label: "add `#[derive]`".to_string(),
            edit: edit.finish(),
            cursor_position: Some(offset),
        }
    });

    // Insert `derive` after doc comments.
    fn derive_insertion_offset(nominal: ast::NominalDef) -> Option<TextUnit> {
        let non_ws_child = nominal
            .syntax()
            .children()
            .find(|it| it.kind() != COMMENT && it.kind() != WHITESPACE)?;
        Some(non_ws_child.range().start())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::check_action;

    #[test]
    fn add_derive_new() {
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
    }

    #[test]
    fn add_derive_existing() {
        check_action(
            "#[derive(Clone)]\nstruct Foo { a: i32<|>, }",
            "#[derive(Clone<|>)]\nstruct Foo { a: i32, }",
            |file, off| add_derive(file, off).map(|f| f()),
        );
    }

    #[test]
    fn add_derive_new_with_doc_comment() {
        check_action(
            "
/// `Foo` is a pretty important struct.
/// It does stuff.
struct Foo { a: i32<|>, }
            ",
            "
/// `Foo` is a pretty important struct.
/// It does stuff.
#[derive(<|>)]
struct Foo { a: i32, }
            ",
            |file, off| add_derive(file, off).map(|f| f()),
        );
    }
}
