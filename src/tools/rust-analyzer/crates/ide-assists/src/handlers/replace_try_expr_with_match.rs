use std::iter;

use ide_db::{
    assists::{AssistId, AssistKind},
    ty_filter::TryEnum,
};
use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
    AstNode, T,
};

use crate::assist_context::{AssistContext, Assists};

// Assist: replace_try_expr_with_match
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
pub(crate) fn replace_try_expr_with_match(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let qm_kw = ctx.find_token_syntax_at_offset(T![?])?;
    let qm_kw_parent = qm_kw.parent().and_then(ast::TryExpr::cast)?;

    let expr = qm_kw_parent.expr()?;
    let expr_type_info = ctx.sema.type_of_expr(&expr)?;

    let try_enum = TryEnum::from_ty(&ctx.sema, &expr_type_info.original)?;

    let target = qm_kw_parent.syntax().text_range();
    acc.add(
        AssistId("replace_try_expr_with_match", AssistKind::RefactorRewrite),
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
                TryEnum::Result => make::expr_return(Some(make::expr_call(
                    make::expr_path(make::ext::ident_path("Err")),
                    make::arg_list(iter::once(make::expr_path(make::ext::ident_path("err")))),
                ))),
            };

            let happy_arm = make::match_arm(
                iter::once(
                    try_enum.happy_pattern(make::ident_pat(false, false, make::name("it")).into()),
                ),
                None,
                make::expr_path(make::ext::ident_path("it")),
            );
            let sad_arm = make::match_arm(iter::once(sad_pat), None, sad_expr);

            let match_arm_list = make::match_arm_list([happy_arm, sad_arm]);

            let expr_match = make::expr_match(expr, match_arm_list)
                .indent(IndentLevel::from_node(qm_kw_parent.syntax()));
            edit.replace_ast::<ast::Expr>(qm_kw_parent.into(), expr_match);
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_replace_try_expr_with_match_not_applicable() {
        check_assist_not_applicable(
            replace_try_expr_with_match,
            r#"
                fn test() {
                    let pat: u32 = 25$0;
                }
            "#,
        );
    }

    #[test]
    fn test_replace_try_expr_with_match_option() {
        check_assist(
            replace_try_expr_with_match,
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
    fn test_replace_try_expr_with_match_result() {
        check_assist(
            replace_try_expr_with_match,
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
}
