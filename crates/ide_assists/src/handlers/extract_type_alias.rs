use syntax::{
    ast::{self, edit::IndentLevel, AstNode},
    match_ast,
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
// type $0Type = (u8, u8, u8);
//
// struct S {
//     field: Type,
// }
// ```
pub(crate) fn extract_type_alias(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    if ctx.frange.range.is_empty() {
        return None;
    }

    let node = ctx.find_node_at_range::<ast::Type>()?;
    let item = ctx.find_node_at_offset::<ast::Item>()?;
    let insert = match_ast! {
        match (item.syntax().parent()?) {
            ast::AssocItemList(it) => it.syntax().parent()?,
            _ => item.syntax().clone(),
        }
    };
    let indent = IndentLevel::from_node(&insert);
    let insert = insert.text_range().start();
    let target = node.syntax().text_range();

    acc.add(
        AssistId("extract_type_alias", AssistKind::RefactorExtract),
        "Extract type as type alias",
        target,
        |builder| {
            builder.edit_file(ctx.frange.file_id);
            builder.replace(target, "Type");
            match ctx.config.snippet_cap {
                Some(cap) => {
                    builder.insert_snippet(
                        cap,
                        insert,
                        format!("type $0Type = {};\n\n{}", node, indent),
                    );
                }
                None => {
                    builder.insert(insert, format!("type Type = {};\n\n{}", node, indent));
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
type $0Type = u8;

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

type $0Type = ();

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
type $0Type = Vec<u8>;

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
type $0Type = u8;

struct S {
    field: (Type,),
}
            "#,
        );
    }

    #[test]
    fn extract_from_impl_or_trait() {
        // When invoked in an impl/trait, extracted type alias should be placed next to the
        // impl/trait, not inside.
        check_assist(
            extract_type_alias,
            r#"
impl S {
    fn f() -> $0(u8, u8)$0 {}
}
            "#,
            r#"
type $0Type = (u8, u8);

impl S {
    fn f() -> Type {}
}
            "#,
        );
        check_assist(
            extract_type_alias,
            r#"
trait Tr {
    fn f() -> $0(u8, u8)$0 {}
}
            "#,
            r#"
type $0Type = (u8, u8);

trait Tr {
    fn f() -> Type {}
}
            "#,
        );
    }

    #[test]
    fn indentation() {
        check_assist(
            extract_type_alias,
            r#"
mod m {
    fn f() -> $0u8$0 {}
}
            "#,
            r#"
mod m {
    type $0Type = u8;

    fn f() -> Type {}
}
            "#,
        );
    }
}
