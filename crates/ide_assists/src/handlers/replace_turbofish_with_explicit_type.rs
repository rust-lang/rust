use syntax::{
    ast::Expr,
    ast::{LetStmt, Type::InferType},
    AstNode, TextRange,
};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: replace_turbofish_with_explicit_type
//
// Converts `::<_>` to an explicit type assignment.
//
// ```
// fn make<T>() -> T { ) }
// fn main() {
//     let a = make$0::<i32>();
// }
// ```
// ->
// ```
// fn make<T>() -> T { ) }
// fn main() {
//     let a: i32 = make();
// }
// ```
pub(crate) fn replace_turbofish_with_explicit_type(
    acc: &mut Assists,
    ctx: &AssistContext,
) -> Option<()> {
    let let_stmt = ctx.find_node_at_offset::<LetStmt>()?;

    let initializer = let_stmt.initializer()?;

    let (turbofish_start, turbofish_type, turbofish_end) = if let Expr::CallExpr(ce) = initializer {
        if let Expr::PathExpr(pe) = ce.expr()? {
            let path = pe.path()?;

            let generic_args = path.segment()?.generic_arg_list()?;

            let colon2 = generic_args.coloncolon_token()?;
            let r_angle = generic_args.r_angle_token()?;

            let turbofish_args_as_string = generic_args
                .generic_args()
                .into_iter()
                .map(|a| -> String { a.to_string() })
                .collect::<Vec<String>>()
                .join(", ");

            (colon2.text_range().start(), turbofish_args_as_string, r_angle.text_range().end())
        } else {
            cov_mark::hit!(not_applicable_if_non_path_function_call);
            return None;
        }
    } else {
        cov_mark::hit!(not_applicable_if_non_function_call_initializer);
        return None;
    };

    let turbofish_range = TextRange::new(turbofish_start, turbofish_end);

    if let None = let_stmt.colon_token() {
        // If there's no colon in a let statement, then there is no explicit type.
        // let x = fn::<...>();
        let ident_range = let_stmt.pat()?.syntax().text_range();

        return acc.add(
            AssistId("replace_turbofish_with_explicit_type", AssistKind::RefactorRewrite),
            format!("Replace turbofish with explicit type `: <{}>`", turbofish_type),
            turbofish_range,
            |builder| {
                builder.insert(ident_range.end(), format!(": {}", turbofish_type));
                builder.delete(turbofish_range);
            },
        );
    } else if let Some(InferType(t)) = let_stmt.ty() {
        // If there's a type inferrence underscore, we can offer to replace it with the type in
        // the turbofish.
        // let x: _ = fn::<...>();
        let underscore_range = t.syntax().text_range();

        return acc.add(
            AssistId("replace_turbofish_with_explicit_type", AssistKind::RefactorRewrite),
            format!("Replace `_` with turbofish type `{}`", turbofish_type),
            turbofish_range,
            |builder| {
                builder.replace(underscore_range, turbofish_type);
                builder.delete(turbofish_range);
            },
        );
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn replaces_turbofish_for_vec_string() {
        check_assist(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a = make$0::<Vec<String>>();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let a: Vec<String> = make();
}
"#,
        );
    }

    #[test]
    fn replace_turbofish_target() {
        check_assist_target(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a = $0make::<Vec<String>>();
}
"#,
            r#"::<Vec<String>>"#,
        );
    }

    #[test]
    fn replace_inferred_type_placeholder() {
        check_assist(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a: _ = make$0::<Vec<String>>();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let a: Vec<String> = make();
}
"#,
        );
    }

    #[test]
    fn not_applicable_constant_initializer() {
        cov_mark::check!(not_applicable_if_non_function_call_initializer);
        check_assist_not_applicable(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a = "foo"$0;
}
"#,
        );
    }

    #[test]
    fn not_applicable_non_path_function_call() {
        cov_mark::check!(not_applicable_if_non_path_function_call);
        check_assist_not_applicable(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    $0let a = (|| {})();
}
"#,
        );
    }
}
