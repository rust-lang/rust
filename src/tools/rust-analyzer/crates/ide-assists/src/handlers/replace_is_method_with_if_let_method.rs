use either::Either;
use ide_db::syntax_helpers::suggest_name;
use syntax::ast::{self, AstNode, HasArgList, syntax_factory::SyntaxFactory};

use crate::{AssistContext, AssistId, Assists, utils::cover_let_chain};

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
    let has_cond = ctx.find_node_at_offset::<Either<ast::IfExpr, ast::WhileExpr>>()?;

    let cond = either::for_both!(&has_cond, it => it.condition())?;
    let cond = cover_let_chain(cond, ctx.selection_trimmed())?;
    let call_expr = match cond {
        ast::Expr::MethodCallExpr(call) => call,
        _ => return None,
    };

    let token = call_expr.name_ref()?.ident_token()?;
    let method_kind = token.text().strip_suffix("_and").unwrap_or(token.text());
    match method_kind {
        "is_some" | "is_ok" => {
            let receiver = call_expr.receiver()?;

            let mut name_generator = suggest_name::NameGenerator::new_from_scope_locals(
                ctx.sema.scope(has_cond.syntax()),
            );
            let var_name = if let ast::Expr::PathExpr(path_expr) = receiver.clone() {
                name_generator.suggest_name(&path_expr.path()?.to_string())
            } else {
                name_generator.for_variable(&receiver, &ctx.sema)
            };
            let (pat, predicate) = method_predicate(&call_expr).unzip();

            let (assist_id, message, text) = if method_kind == "is_some" {
                ("replace_is_some_with_if_let_some", "Replace `is_some` with `let Some`", "Some")
            } else {
                ("replace_is_ok_with_if_let_ok", "Replace `is_ok` with `let Ok`", "Ok")
            };

            acc.add(
                AssistId::refactor_rewrite(assist_id),
                message,
                call_expr.syntax().text_range(),
                |edit| {
                    let make = SyntaxFactory::with_mappings();
                    let mut editor = edit.make_editor(call_expr.syntax());

                    let var_pat = pat.unwrap_or_else(|| {
                        make.ident_pat(false, false, make.name(&var_name)).into()
                    });
                    let pat = make.tuple_struct_pat(make.ident_path(text), [var_pat]).into();
                    let let_expr = make.expr_let(pat, receiver);

                    if let Some(cap) = ctx.config.snippet_cap
                        && let Some(ast::Pat::TupleStructPat(pat)) = let_expr.pat()
                        && let Some(first_var) = pat.fields().next()
                        && predicate.is_none()
                    {
                        let placeholder = edit.make_placeholder_snippet(cap);
                        editor.add_annotation(first_var.syntax(), placeholder);
                    }

                    let new_expr = if let Some(predicate) = predicate {
                        let op = ast::BinaryOp::LogicOp(ast::LogicOp::And);
                        make.expr_bin(let_expr.into(), op, predicate).into()
                    } else {
                        ast::Expr::from(let_expr)
                    };
                    editor.replace(call_expr.syntax(), new_expr.syntax());

                    editor.add_mappings(make.finish_with_mappings());
                    edit.add_file_edits(ctx.vfs_file_id(), editor);
                },
            )
        }
        _ => None,
    }
}

fn method_predicate(call_expr: &ast::MethodCallExpr) -> Option<(ast::Pat, ast::Expr)> {
    let argument = call_expr.arg_list()?.args().next()?;
    match argument {
        ast::Expr::ClosureExpr(it) => {
            let pat = it.param_list()?.params().next()?.pat()?;
            Some((pat, it.body()?))
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

    #[test]
    fn replace_is_some_and_with_if_let_chain_some_works() {
        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Some(1);
    if x.is_som$0e_and(|it| it != 3) {}
}
"#,
            r#"
fn main() {
    let x = Some(1);
    if let Some(it) = x && it != 3 {}
}
"#,
        );
    }

    #[test]
    fn replace_is_some_with_if_let_some_in_let_chain() {
        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Some(1);
    let cond = true;
    if cond && x.is_som$0e() {}
}
"#,
            r#"
fn main() {
    let x = Some(1);
    let cond = true;
    if cond && let Some(${0:x1}) = x {}
}
"#,
        );

        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Some(1);
    let cond = true;
    if x.is_som$0e() && cond {}
}
"#,
            r#"
fn main() {
    let x = Some(1);
    let cond = true;
    if let Some(${0:x1}) = x && cond {}
}
"#,
        );

        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Some(1);
    let cond = true;
    if cond && x.is_som$0e() && cond {}
}
"#,
            r#"
fn main() {
    let x = Some(1);
    let cond = true;
    if cond && let Some(${0:x1}) = x && cond {}
}
"#,
        );
    }

    #[test]
    fn replace_is_some_with_while_let_some() {
        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let mut x = Some(1);
    while x.is_som$0e() { x = None }
}
"#,
            r#"
fn main() {
    let mut x = Some(1);
    while let Some(${0:x1}) = x { x = None }
}
"#,
        );
    }

    #[test]
    fn replace_is_some_with_if_let_some_not_applicable_after_l_curly() {
        check_assist_not_applicable(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Some(1);
    if x.is_some() {
        ()$0
    }
}
"#,
        );
    }
}
