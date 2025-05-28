use std::iter;

use ide_db::{
    assists::{AssistId, ExprFillDefaultMode},
    ty_filter::TryEnum,
};
use syntax::{
    AstNode, T,
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make,
        syntax_factory::SyntaxFactory,
    },
};

use crate::assist_context::{AssistContext, Assists};

// Assist: desugar_try_expr_match
//
// Replaces a `try` expression with a `match` expression.
//
// ```
// # //- minicore: try, option
// fn handle() {
//     let pat = Some(true)$0?;
// }
// ```
// ->
// ```
// fn handle() {
//     let pat = match Some(true) {
//         Some(it) => it,
//         None => return None,
//     };
// }
// ```

// Assist: desugar_try_expr_let_else
//
// Replaces a `try` expression with a `let else` statement.
//
// ```
// # //- minicore: try, option
// fn handle() {
//     let pat = Some(true)$0?;
// }
// ```
// ->
// ```
// fn handle() {
//     let Some(pat) = Some(true) else {
//         return None;
//     };
// }
// ```
pub(crate) fn desugar_try_expr(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let question_tok = ctx.find_token_syntax_at_offset(T![?])?;
    let try_expr = question_tok.parent().and_then(ast::TryExpr::cast)?;

    let expr = try_expr.expr()?;
    let expr_type_info = ctx.sema.type_of_expr(&expr)?;

    let try_enum = TryEnum::from_ty(&ctx.sema, &expr_type_info.original)?;

    let target = try_expr.syntax().text_range();
    acc.add(
        AssistId::refactor_rewrite("desugar_try_expr_match"),
        "Replace try expression with match",
        target,
        |edit| {
            let sad_pat = match try_enum {
                TryEnum::Option => make::path_pat(make::ext::ident_path("None")),
                TryEnum::Result => make::tuple_struct_pat(
                    make::ext::ident_path("Err"),
                    iter::once(make::path_pat(make::ext::ident_path("err"))),
                )
                .into(),
            };
            let sad_expr = match try_enum {
                TryEnum::Option => {
                    make::expr_return(Some(make::expr_path(make::ext::ident_path("None"))))
                }
                TryEnum::Result => make::expr_return(Some(
                    make::expr_call(
                        make::expr_path(make::ext::ident_path("Err")),
                        make::arg_list(iter::once(make::expr_path(make::ext::ident_path("err")))),
                    )
                    .into(),
                )),
            };

            let happy_arm = make::match_arm(
                try_enum.happy_pattern(make::ident_pat(false, false, make::name("it")).into()),
                None,
                make::expr_path(make::ext::ident_path("it")),
            );
            let sad_arm = make::match_arm(sad_pat, None, sad_expr);

            let match_arm_list = make::match_arm_list([happy_arm, sad_arm]);

            let expr_match = make::expr_match(expr.clone(), match_arm_list)
                .indent(IndentLevel::from_node(try_expr.syntax()));

            edit.replace_ast::<ast::Expr>(try_expr.clone().into(), expr_match.into());
        },
    );

    if let Some(let_stmt) = try_expr.syntax().parent().and_then(ast::LetStmt::cast) {
        if let_stmt.let_else().is_none() {
            let pat = let_stmt.pat()?;
            acc.add(
                AssistId::refactor_rewrite("desugar_try_expr_let_else"),
                "Replace try expression with let else",
                target,
                |builder| {
                    let make = SyntaxFactory::with_mappings();
                    let mut editor = builder.make_editor(let_stmt.syntax());

                    let indent_level = IndentLevel::from_node(let_stmt.syntax());
                    let new_let_stmt = make.let_else_stmt(
                        try_enum.happy_pattern(pat),
                        let_stmt.ty(),
                        expr,
                        make.block_expr(
                            iter::once(
                                make.expr_stmt(
                                    make.expr_return(Some(match try_enum {
                                        TryEnum::Option => make.expr_path(make.ident_path("None")),
                                        TryEnum::Result => make
                                            .expr_call(
                                                make.expr_path(make.ident_path("Err")),
                                                make.arg_list(iter::once(
                                                    match ctx.config.expr_fill_default {
                                                        ExprFillDefaultMode::Todo => make
                                                            .expr_macro(
                                                                make.ident_path("todo"),
                                                                make.token_tree(
                                                                    syntax::SyntaxKind::L_PAREN,
                                                                    [],
                                                                ),
                                                            )
                                                            .into(),
                                                        ExprFillDefaultMode::Underscore => {
                                                            make.expr_underscore().into()
                                                        }
                                                        ExprFillDefaultMode::Default => make
                                                            .expr_macro(
                                                                make.ident_path("todo"),
                                                                make.token_tree(
                                                                    syntax::SyntaxKind::L_PAREN,
                                                                    [],
                                                                ),
                                                            )
                                                            .into(),
                                                    },
                                                )),
                                            )
                                            .into(),
                                    }))
                                    .indent(indent_level + 1)
                                    .into(),
                                )
                                .into(),
                            ),
                            None,
                        )
                        .indent(indent_level),
                    );
                    editor.replace(let_stmt.syntax(), new_let_stmt.syntax());
                    editor.add_mappings(make.finish_with_mappings());
                    builder.add_file_edits(ctx.vfs_file_id(), editor);
                },
            );
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_by_label, check_assist_not_applicable};

    #[test]
    fn test_desugar_try_expr_not_applicable() {
        check_assist_not_applicable(
            desugar_try_expr,
            r#"
                fn test() {
                    let pat: u32 = 25$0;
                }
            "#,
        );
    }

    #[test]
    fn test_desugar_try_expr_option() {
        check_assist(
            desugar_try_expr,
            r#"
//- minicore: try, option
fn test() {
    let pat = Some(true)$0?;
}
            "#,
            r#"
fn test() {
    let pat = match Some(true) {
        Some(it) => it,
        None => return None,
    };
}
            "#,
        );
    }

    #[test]
    fn test_desugar_try_expr_result() {
        check_assist(
            desugar_try_expr,
            r#"
//- minicore: try, from, result
fn test() {
    let pat = Ok(true)$0?;
}
            "#,
            r#"
fn test() {
    let pat = match Ok(true) {
        Ok(it) => it,
        Err(err) => return Err(err),
    };
}
            "#,
        );
    }

    #[test]
    fn test_desugar_try_expr_option_let_else() {
        check_assist_by_label(
            desugar_try_expr,
            r#"
//- minicore: try, option
fn test() {
    let pat = Some(true)$0?;
}
            "#,
            r#"
fn test() {
    let Some(pat) = Some(true) else {
        return None;
    };
}
            "#,
            "Replace try expression with let else",
        );
    }

    #[test]
    fn test_desugar_try_expr_result_let_else() {
        check_assist_by_label(
            desugar_try_expr,
            r#"
//- minicore: try, from, result
fn test() {
    let pat = Ok(true)$0?;
}
            "#,
            r#"
fn test() {
    let Ok(pat) = Ok(true) else {
        return Err(todo!());
    };
}
            "#,
            "Replace try expression with let else",
        );
    }
}
