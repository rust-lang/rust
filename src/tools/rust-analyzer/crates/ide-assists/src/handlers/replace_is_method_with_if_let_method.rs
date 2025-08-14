use ide_db::syntax_helpers::suggest_name;
use syntax::ast::{self, AstNode, syntax_factory::SyntaxFactory};

use crate::{AssistContext, AssistId, Assists};

// Assist: replace_is_some_with_if_let_some
//
// Replace `if x.is_some()` with `if let Some(_tmp) = x` or `if x.is_ok()` with `if let Ok(_tmp) = x`.
//
// ```
// fn main() {
//     let x = Some(1);
//     if x.is_som$0e() {}
// }
// ```
// ->
// ```
// fn main() {
//     let x = Some(1);
//     if let Some(${0:x1}) = x {}
// }
// ```
pub(crate) fn replace_is_method_with_if_let_method(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let if_expr = ctx.find_node_at_offset::<ast::IfExpr>()?;

    let cond = if_expr.condition()?;
    let call_expr = match cond {
        ast::Expr::MethodCallExpr(call) => call,
        _ => return None,
    };

    let name_ref = call_expr.name_ref()?;
    match name_ref.text().as_str() {
        "is_some" | "is_ok" => {
            let receiver = call_expr.receiver()?;

            let mut name_generator = suggest_name::NameGenerator::new_from_scope_locals(
                ctx.sema.scope(if_expr.syntax()),
            );
            let var_name = if let ast::Expr::PathExpr(path_expr) = receiver.clone() {
                name_generator.suggest_name(&path_expr.path()?.to_string())
            } else {
                name_generator.for_variable(&receiver, &ctx.sema)
            };

            let (assist_id, message, text) = if name_ref.text() == "is_some" {
                ("replace_is_some_with_if_let_some", "Replace `is_some` with `if let Some`", "Some")
            } else {
                ("replace_is_ok_with_if_let_ok", "Replace `is_ok` with `if let Ok`", "Ok")
            };

            acc.add(
                AssistId::refactor_rewrite(assist_id),
                message,
                call_expr.syntax().text_range(),
                |edit| {
                    let make = SyntaxFactory::with_mappings();
                    let mut editor = edit.make_editor(call_expr.syntax());

                    let var_pat = make.ident_pat(false, false, make.name(&var_name));
                    let pat = make.tuple_struct_pat(make.ident_path(text), [var_pat.into()]);
                    let let_expr = make.expr_let(pat.into(), receiver);

                    if let Some(cap) = ctx.config.snippet_cap
                        && let Some(ast::Pat::TupleStructPat(pat)) = let_expr.pat()
                        && let Some(first_var) = pat.fields().next()
                    {
                        let placeholder = edit.make_placeholder_snippet(cap);
                        editor.add_annotation(first_var.syntax(), placeholder);
                    }

                    editor.replace(call_expr.syntax(), let_expr.syntax());
                    editor.add_mappings(make.finish_with_mappings());
                    edit.add_file_edits(ctx.vfs_file_id(), editor);
                },
            )
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::replace_is_method_with_if_let_method;

    #[test]
    fn replace_is_some_with_if_let_some_works() {
        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Some(1);
    if x.is_som$0e() {}
}
"#,
            r#"
fn main() {
    let x = Some(1);
    if let Some(${0:x1}) = x {}
}
"#,
        );

        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn test() -> Option<i32> {
    Some(1)
}
fn main() {
    if test().is_som$0e() {}
}
"#,
            r#"
fn test() -> Option<i32> {
    Some(1)
}
fn main() {
    if let Some(${0:test}) = test() {}
}
"#,
        );
    }

    #[test]
    fn replace_is_some_with_if_let_some_not_applicable() {
        check_assist_not_applicable(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Some(1);
    if x.is_non$0e() {}
}
"#,
        );
    }

    #[test]
    fn replace_is_ok_with_if_let_ok_works() {
        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Ok(1);
    if x.is_o$0k() {}
}
"#,
            r#"
fn main() {
    let x = Ok(1);
    if let Ok(${0:x1}) = x {}
}
"#,
        );

        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn test() -> Result<i32> {
    Ok(1)
}
fn main() {
    if test().is_o$0k() {}
}
"#,
            r#"
fn test() -> Result<i32> {
    Ok(1)
}
fn main() {
    if let Ok(${0:test}) = test() {}
}
"#,
        );
    }

    #[test]
    fn replace_is_ok_with_if_let_ok_not_applicable() {
        check_assist_not_applicable(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Ok(1);
    if x.is_e$0rr() {}
}
"#,
        );
    }
}
