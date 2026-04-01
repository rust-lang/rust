use ide_db::assists::AssistId;
use syntax::{
    AstNode,
    ast::{self, GenericArg, HasGenericArgs},
};

use crate::{AssistContext, Assists};

// Assist: unwrap_type_to_generic_arg
//
// This assist unwraps a type into its generic type argument.
//
// ```
// fn foo() -> $0Option<i32> {
//     todo!()
// }
// ```
// ->
// ```
// fn foo() -> i32 {
//     todo!()
// }
// ```
pub(crate) fn unwrap_type_to_generic_arg(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let path_type = ctx.find_node_at_offset::<ast::PathType>()?;
    let path = path_type.path()?;
    let segment = path.segment()?;
    let args_list = segment.generic_arg_list()?;

    let mut generic_arg = None;

    for arg in args_list.generic_args() {
        match arg {
            GenericArg::ConstArg(_) | GenericArg::LifetimeArg(_) => (),
            GenericArg::TypeArg(arg) if generic_arg.is_none() => {
                generic_arg = Some(arg);
            }
            _ => return None,
        }
    }

    let generic_arg = generic_arg?;

    acc.add(
        AssistId::refactor_extract("unwrap_type_to_generic_arg"),
        format!("Unwrap type to type argument {generic_arg}"),
        path_type.syntax().text_range(),
        |builder| {
            let mut editor = builder.make_editor(path_type.syntax());
            editor.replace(path_type.syntax(), generic_arg.syntax());

            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_unwrap_type_to_generic_arg() {
        check_assist(
            unwrap_type_to_generic_arg,
            r#"
//- minicore: option
fn foo() -> $0Option<i32> {
    todo!()
}
"#,
            r#"
fn foo() -> i32 {
    todo!()
}
"#,
        );
    }

    #[test]
    fn unwrap_type_to_generic_arg_not_applicable_for_non_generic_arg_list() {
        check_assist_not_applicable(
            unwrap_type_to_generic_arg,
            r#"
fn foo() -> $0i32 {}
"#,
        );
    }

    #[test]
    fn unwrap_type_to_generic_arg_not_applicable_for_multiple_generic_args() {
        check_assist_not_applicable(
            unwrap_type_to_generic_arg,
            r#"
//- minicore: result
fn foo() -> $0Result<i32, ()> {
    todo!()
}
"#,
        );
    }

    #[test]
    fn unwrap_type_to_generic_arg_with_lifetime_and_const() {
        check_assist(
            unwrap_type_to_generic_arg,
            r#"
enum Foo<'a, T, const N: usize> {
    Bar(T),
    Baz(&'a [T; N]),
}

fn test<'a>() -> $0Foo<'a, i32, 3> {
    todo!()
}
"#,
            r#"
enum Foo<'a, T, const N: usize> {
    Bar(T),
    Baz(&'a [T; N]),
}

fn test<'a>() -> i32 {
    todo!()
}
"#,
        );
    }

    #[test]
    fn unwrap_type_to_generic_arg_in_let_stmt() {
        check_assist(
            unwrap_type_to_generic_arg,
            r#"
enum Foo<T> {
    Bar(T),
    Baz,
}

fn test() {
    let foo: $0Foo<i32> = todo!();
}
"#,
            r#"
enum Foo<T> {
    Bar(T),
    Baz,
}

fn test() {
    let foo: i32 = todo!();
}
"#,
        );
    }
}
