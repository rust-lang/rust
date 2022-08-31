use syntax::{
    ast::{Expr, GenericArg},
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
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let let_stmt = ctx.find_node_at_offset::<LetStmt>()?;

    let initializer = let_stmt.initializer()?;

    let generic_args = match &initializer {
        Expr::MethodCallExpr(ce) => ce.generic_arg_list()?,
        Expr::CallExpr(ce) => {
            if let Expr::PathExpr(pe) = ce.expr()? {
                pe.path()?.segment()?.generic_arg_list()?
            } else {
                cov_mark::hit!(not_applicable_if_non_path_function_call);
                return None;
            }
        }
        _ => {
            cov_mark::hit!(not_applicable_if_non_function_call_initializer);
            return None;
        }
    };

    // Find range of ::<_>
    let colon2 = generic_args.coloncolon_token()?;
    let r_angle = generic_args.r_angle_token()?;
    let turbofish_range = TextRange::new(colon2.text_range().start(), r_angle.text_range().end());

    let turbofish_args: Vec<GenericArg> = generic_args.generic_args().into_iter().collect();

    // Find type of ::<_>
    if turbofish_args.len() != 1 {
        cov_mark::hit!(not_applicable_if_not_single_arg);
        return None;
    }

    // An improvement would be to check that this is correctly part of the return value of the
    // function call, or sub in the actual return type.
    let turbofish_type = &turbofish_args[0];

    let initializer_start = initializer.syntax().text_range().start();
    if ctx.offset() > turbofish_range.end() || ctx.offset() < initializer_start {
        cov_mark::hit!(not_applicable_outside_turbofish);
        return None;
    }

    if let None = let_stmt.colon_token() {
        // If there's no colon in a let statement, then there is no explicit type.
        // let x = fn::<...>();
        let ident_range = let_stmt.pat()?.syntax().text_range();

        return acc.add(
            AssistId("replace_turbofish_with_explicit_type", AssistKind::RefactorRewrite),
            "Replace turbofish with explicit type",
            TextRange::new(initializer_start, turbofish_range.end()),
            |builder| {
                builder.insert(ident_range.end(), format!(": {}", turbofish_type));
                builder.delete(turbofish_range);
            },
        );
    } else if let Some(InferType(t)) = let_stmt.ty() {
        // If there's a type inference underscore, we can offer to replace it with the type in
        // the turbofish.
        // let x: _ = fn::<...>();
        let underscore_range = t.syntax().text_range();

        return acc.add(
            AssistId("replace_turbofish_with_explicit_type", AssistKind::RefactorRewrite),
            "Replace `_` with turbofish type",
            turbofish_range,
            |builder| {
                builder.replace(underscore_range, turbofish_type.to_string());
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
    fn replaces_method_calls() {
        // foo.make() is a method call which uses a different expr in the let initializer
        check_assist(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a = foo.make$0::<Vec<String>>();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let a: Vec<String> = foo.make();
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
            r#"make::<Vec<String>>"#,
        );
    }

    #[test]
    fn not_applicable_outside_turbofish() {
        cov_mark::check!(not_applicable_outside_turbofish);
        check_assist_not_applicable(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let $0a = make::<Vec<String>>();
}
"#,
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

    #[test]
    fn non_applicable_multiple_generic_args() {
        cov_mark::check!(not_applicable_if_not_single_arg);
        check_assist_not_applicable(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a = make$0::<Vec<String>, i32>();
}
"#,
        );
    }
}
