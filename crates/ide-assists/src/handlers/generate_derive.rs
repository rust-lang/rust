use syntax::{
    ast::{self, edit::IndentLevel, AstNode, HasAttrs},
    SyntaxKind::{COMMENT, WHITESPACE},
    TextSize,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: generate_derive
//
// Adds a new `#[derive()]` clause to a struct or enum.
//
// ```
// struct Point {
//     x: u32,
//     y: u32,$0
// }
// ```
// ->
// ```
// #[derive($0)]
// struct Point {
//     x: u32,
//     y: u32,
// }
// ```
pub(crate) fn generate_derive(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let cap = ctx.config.snippet_cap?;
    let nominal = ctx.find_node_at_offset::<ast::Adt>()?;
    let node_start = derive_insertion_offset(&nominal)?;
    let target = nominal.syntax().text_range();
    acc.add(
        AssistId("generate_derive", AssistKind::Generate),
        "Add `#[derive]`",
        target,
        |builder| {
            let derive_attr = nominal
                .attrs()
                .filter_map(|x| x.as_simple_call())
                .filter(|(name, _arg)| name == "derive")
                .map(|(_name, arg)| arg)
                .next();
            match derive_attr {
                None => {
                    let indent_level = IndentLevel::from_node(nominal.syntax());
                    builder.insert_snippet(
                        cap,
                        node_start,
                        format!("#[derive($0)]\n{indent_level}"),
                    );
                }
                Some(tt) => {
                    // Just move the cursor.
                    builder.insert_snippet(
                        cap,
                        tt.syntax().text_range().end() - TextSize::of(')'),
                        "$0",
                    )
                }
            };
        },
    )
}

// Insert `derive` after doc comments.
fn derive_insertion_offset(nominal: &ast::Adt) -> Option<TextSize> {
    let non_ws_child = nominal
        .syntax()
        .children_with_tokens()
        .find(|it| it.kind() != COMMENT && it.kind() != WHITESPACE)?;
    Some(non_ws_child.text_range().start())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_target};

    use super::*;

    #[test]
    fn add_derive_new() {
        check_assist(
            generate_derive,
            "struct Foo { a: i32, $0}",
            "#[derive($0)]\nstruct Foo { a: i32, }",
        );
        check_assist(
            generate_derive,
            "struct Foo { $0 a: i32, }",
            "#[derive($0)]\nstruct Foo {  a: i32, }",
        );
        check_assist(
            generate_derive,
            "
mod m {
    struct Foo { a: i32,$0 }
}
            ",
            "
mod m {
    #[derive($0)]
    struct Foo { a: i32, }
}
            ",
        );
    }

    #[test]
    fn add_derive_existing() {
        check_assist(
            generate_derive,
            "#[derive(Clone)]\nstruct Foo { a: i32$0, }",
            "#[derive(Clone$0)]\nstruct Foo { a: i32, }",
        );
    }

    #[test]
    fn add_derive_new_with_doc_comment() {
        check_assist(
            generate_derive,
            "
/// `Foo` is a pretty important struct.
/// It does stuff.
struct Foo { a: i32$0, }
            ",
            "
/// `Foo` is a pretty important struct.
/// It does stuff.
#[derive($0)]
struct Foo { a: i32, }
            ",
        );
        check_assist(
            generate_derive,
            "
mod m {
    /// `Foo` is a pretty important struct.
    /// It does stuff.
    struct Foo { a: i32,$0 }
}
            ",
            "
mod m {
    /// `Foo` is a pretty important struct.
    /// It does stuff.
    #[derive($0)]
    struct Foo { a: i32, }
}
            ",
        );
    }

    #[test]
    fn add_derive_target() {
        check_assist_target(
            generate_derive,
            "
struct SomeThingIrrelevant;
/// `Foo` is a pretty important struct.
/// It does stuff.
struct Foo { a: i32$0, }
struct EvenMoreIrrelevant;
            ",
            "/// `Foo` is a pretty important struct.
/// It does stuff.
struct Foo { a: i32, }",
        );
    }
}
