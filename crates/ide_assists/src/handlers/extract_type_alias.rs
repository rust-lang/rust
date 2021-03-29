use syntax::ast::{self, AstNode};

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
// type ${0:Type} = (u8, u8, u8);
//
// struct S {
//     field: ${0:Type},
// }
// ```
pub(crate) fn extract_type_alias(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    if ctx.frange.range.is_empty() {
        return None;
    }

    let node = ctx.find_node_at_range::<ast::Type>()?;
    let insert = ctx.find_node_at_offset::<ast::Item>()?.syntax().text_range().start();
    let target = node.syntax().text_range();

    acc.add(
        AssistId("extract_type_alias", AssistKind::RefactorExtract),
        "Extract type as type alias",
        target,
        |builder| {
            builder.edit_file(ctx.frange.file_id);
            match ctx.config.snippet_cap {
                Some(cap) => {
                    builder.replace_snippet(cap, target, "${0:Type}");
                    builder.insert_snippet(
                        cap,
                        insert,
                        format!("type ${{0:Type}} = {};\n\n", node),
                    );
                }
                None => {
                    builder.replace(target, "Type");
                    builder.insert(insert, format!("type Type = {};\n\n", node));
                }
            }
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
type ${0:Type} = u8;

struct S {
    field: ${0:Type},
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

type ${0:Type} = ();

fn f() {
    generic::<${0:Type}>();
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
type ${0:Type} = Vec<u8>;

struct S {
    v: Vec<Vec<${0:Type}>>,
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
type ${0:Type} = u8;

struct S {
    field: (${0:Type},),
}
            "#,
        );
    }
}
