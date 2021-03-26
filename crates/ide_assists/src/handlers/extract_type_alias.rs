use syntax::{
    ast::{self, AstNode},
    SyntaxKind,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: extract_type_alias
//
// Extracts the selected type as a type alias.
//
// ```
// struct S {
//     field: $0(u8, u8, u8)$0,
// }
// ```
// ->
// ```
// type Type = (u8, u8, u8);
//
// struct S {
//     field: Type,
// }
// ```
pub(crate) fn extract_type_alias(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    if ctx.frange.range.is_empty() {
        return None;
    }

    let node = match ctx.covering_element() {
        syntax::NodeOrToken::Node(node) => node,
        syntax::NodeOrToken::Token(tok) => tok.parent()?,
    };
    let range = node.text_range();
    let mut type_like_node = None;
    for node in node.ancestors() {
        if node.text_range() != range {
            break;
        }

        let kind = node.kind();
        if ast::Type::can_cast(kind) || kind == SyntaxKind::TYPE_ARG {
            type_like_node = Some(node);
            break;
        }
    }

    let node = type_like_node?;

    let insert = ctx.find_node_at_offset::<ast::Item>()?.syntax().text_range().start();
    let target = node.text_range();

    acc.add(
        AssistId("extract_type_alias", AssistKind::RefactorExtract),
        "Extract type as type alias",
        target,
        |builder| {
            builder.edit_file(ctx.frange.file_id);
            // FIXME: add snippet support
            builder.replace(target, "Type");
            builder.insert(insert, format!("type Type = {};\n\n", node));
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_not_applicable_without_selection() {
        check_assist_not_applicable(
            extract_type_alias,
            r"
struct S {
    field: $0(u8, u8, u8),
}
            ",
        );
    }

    #[test]
    fn test_simple_types() {
        check_assist(
            extract_type_alias,
            r"
struct S {
    field: $0u8$0,
}
            ",
            r#"
type Type = u8;

struct S {
    field: Type,
}
            "#,
        );
    }

    #[test]
    fn test_generic_type_arg() {
        check_assist(
            extract_type_alias,
            r"
fn generic<T>() {}

fn f() {
    generic::<$0()$0>();
}
            ",
            r#"
fn generic<T>() {}

type Type = ();

fn f() {
    generic::<Type>();
}
            "#,
        );
    }

    #[test]
    fn test_inner_type_arg() {
        check_assist(
            extract_type_alias,
            r"
struct Vec<T> {}
struct S {
    v: Vec<Vec<$0Vec<u8>$0>>,
}
            ",
            r#"
struct Vec<T> {}
type Type = Vec<u8>;

struct S {
    v: Vec<Vec<Type>>,
}
            "#,
        );
    }

    #[test]
    fn test_extract_inner_type() {
        check_assist(
            extract_type_alias,
            r"
struct S {
    field: ($0u8$0,),
}
            ",
            r#"
type Type = u8;

struct S {
    field: (Type,),
}
            "#,
        );
    }
}
