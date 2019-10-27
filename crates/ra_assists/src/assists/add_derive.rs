use hir::db::HirDatabase;
use ra_syntax::{
    ast::{self, AstNode, AttrsOwner},
    SyntaxKind::{COMMENT, WHITESPACE},
    TextUnit,
};

use crate::{Assist, AssistCtx, AssistId};

// Assist: add_derive
//
// Adds a new `#[derive()]` clause to a struct or enum.
//
// ```
// struct Point {
//     x: u32,
//     y: u32,<|>
// }
// ```
// ->
// ```
// #[derive()]
// struct Point {
//     x: u32,
//     y: u32,
// }
// ```
pub(crate) fn add_derive(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let nominal = ctx.find_node_at_offset::<ast::NominalDef>()?;
    let node_start = derive_insertion_offset(&nominal)?;
    ctx.add_assist(AssistId("add_derive"), "add `#[derive]`", |edit| {
        let derive_attr = nominal
            .attrs()
            .filter_map(|x| x.as_simple_call())
            .filter(|(name, _arg)| name == "derive")
            .map(|(_name, arg)| arg)
            .next();
        let offset = match derive_attr {
            None => {
                edit.insert(node_start, "#[derive()]\n");
                node_start + TextUnit::of_str("#[derive(")
            }
            Some(tt) => tt.syntax().text_range().end() - TextUnit::of_char(')'),
        };
        edit.target(nominal.syntax().text_range());
        edit.set_cursor(offset)
    })
}

// Insert `derive` after doc comments.
fn derive_insertion_offset(nominal: &ast::NominalDef) -> Option<TextUnit> {
    let non_ws_child = nominal
        .syntax()
        .children_with_tokens()
        .find(|it| it.kind() != COMMENT && it.kind() != WHITESPACE)?;
    Some(non_ws_child.text_range().start())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{check_assist, check_assist_target};

    #[test]
    fn add_derive_new() {
        check_assist(
            add_derive,
            "struct Foo { a: i32, <|>}",
            "#[derive(<|>)]\nstruct Foo { a: i32, }",
        );
        check_assist(
            add_derive,
            "struct Foo { <|> a: i32, }",
            "#[derive(<|>)]\nstruct Foo {  a: i32, }",
        );
    }

    #[test]
    fn add_derive_existing() {
        check_assist(
            add_derive,
            "#[derive(Clone)]\nstruct Foo { a: i32<|>, }",
            "#[derive(Clone<|>)]\nstruct Foo { a: i32, }",
        );
    }

    #[test]
    fn add_derive_new_with_doc_comment() {
        check_assist(
            add_derive,
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
        );
    }

    #[test]
    fn add_derive_target() {
        check_assist_target(
            add_derive,
            "
struct SomeThingIrrelevant;
/// `Foo` is a pretty important struct.
/// It does stuff.
struct Foo { a: i32<|>, }
struct EvenMoreIrrelevant;
            ",
            "/// `Foo` is a pretty important struct.
/// It does stuff.
struct Foo { a: i32, }",
        );
    }
}
