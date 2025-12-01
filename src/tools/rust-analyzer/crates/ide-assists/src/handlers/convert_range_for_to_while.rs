use ide_db::assists::AssistId;
use itertools::Itertools;
use syntax::{
    AstNode, T,
    algo::previous_non_trivia_token,
    ast::{
        self, HasArgList, HasLoopBody, HasName, RangeItem, edit::AstNodeEdit, make,
        syntax_factory::SyntaxFactory,
    },
    syntax_editor::{Element, Position},
};

use crate::assist_context::{AssistContext, Assists};

// Assist: convert_range_for_to_while
//
// Convert for each range into while loop.
//
// ```
// fn foo() {
//     $0for i in 3..7 {
//         foo(i);
//     }
// }
// ```
// ->
// ```
// fn foo() {
//     let mut i = 3;
//     while i < 7 {
//         foo(i);
//         i += 1;
//     }
// }
// ```
pub(crate) fn convert_range_for_to_while(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let for_kw = ctx.find_token_syntax_at_offset(T![for])?;
    let for_ = ast::ForExpr::cast(for_kw.parent()?)?;
    let ast::Pat::IdentPat(pat) = for_.pat()? else { return None };
    let iterable = for_.iterable()?;
    let (start, end, step, inclusive) = extract_range(&iterable)?;
    let name = pat.name()?;
    let body = for_.loop_body()?;
    let last = previous_non_trivia_token(body.stmt_list()?.r_curly_token()?)?;

    let description = if end.is_some() {
        "Replace with while expression"
    } else {
        "Replace with loop expression"
    };
    acc.add(
        AssistId::refactor("convert_range_for_to_while"),
        description,
        for_.syntax().text_range(),
        |builder| {
            let mut edit = builder.make_editor(for_.syntax());
            let make = SyntaxFactory::with_mappings();

            let indent = for_.indent_level();
            let pat = make.ident_pat(pat.ref_token().is_some(), true, name.clone());
            let let_stmt = make.let_stmt(pat.into(), None, Some(start));
            edit.insert_all(
                Position::before(for_.syntax()),
                vec![
                    let_stmt.syntax().syntax_element(),
                    make.whitespace(&format!("\n{}", indent)).syntax_element(),
                ],
            );

            let mut elements = vec![];

            let var_expr = make.expr_path(make.ident_path(&name.text()));
            let op = ast::BinaryOp::CmpOp(ast::CmpOp::Ord {
                ordering: ast::Ordering::Less,
                strict: !inclusive,
            });
            if let Some(end) = end {
                elements.extend([
                    make.token(T![while]).syntax_element(),
                    make.whitespace(" ").syntax_element(),
                    make.expr_bin(var_expr.clone(), op, end).syntax().syntax_element(),
                ]);
            } else {
                elements.push(make.token(T![loop]).syntax_element());
            }

            edit.replace_all(
                for_kw.syntax_element()..=iterable.syntax().syntax_element(),
                elements,
            );

            let op = ast::BinaryOp::Assignment { op: Some(ast::ArithOp::Add) };
            edit.insert_all(
                Position::after(last),
                vec![
                    make.whitespace(&format!("\n{}", indent + 1)).syntax_element(),
                    make.expr_bin(var_expr, op, step).syntax().syntax_element(),
                    make.token(T![;]).syntax_element(),
                ],
            );

            edit.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), edit);
        },
    )
}

fn extract_range(iterable: &ast::Expr) -> Option<(ast::Expr, Option<ast::Expr>, ast::Expr, bool)> {
    Some(match iterable {
        ast::Expr::ParenExpr(expr) => extract_range(&expr.expr()?)?,
        ast::Expr::RangeExpr(range) => {
            let inclusive = range.op_kind()? == ast::RangeOp::Inclusive;
            (range.start()?, range.end(), make::expr_literal("1").into(), inclusive)
        }
        ast::Expr::MethodCallExpr(call) if call.name_ref()?.text() == "step_by" => {
            let [step] = Itertools::collect_array(call.arg_list()?.args())?;
            let (start, end, _, inclusive) = extract_range(&call.receiver()?)?;
            (start, end, step, inclusive)
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_convert_range_for_to_while() {
        check_assist(
            convert_range_for_to_while,
            "
fn foo() {
    $0for i in 3..7 {
        foo(i);
    }
}
            ",
            "
fn foo() {
    let mut i = 3;
    while i < 7 {
        foo(i);
        i += 1;
    }
}
            ",
        );
    }

    #[test]
    fn test_convert_range_for_to_while_no_end_bound() {
        check_assist(
            convert_range_for_to_while,
            "
fn foo() {
    $0for i in 3.. {
        foo(i);
    }
}
            ",
            "
fn foo() {
    let mut i = 3;
    loop {
        foo(i);
        i += 1;
    }
}
            ",
        );
    }

    #[test]
    fn test_convert_range_for_to_while_with_mut_binding() {
        check_assist(
            convert_range_for_to_while,
            "
fn foo() {
    $0for mut i in 3..7 {
        foo(i);
    }
}
            ",
            "
fn foo() {
    let mut i = 3;
    while i < 7 {
        foo(i);
        i += 1;
    }
}
            ",
        );
    }

    #[test]
    fn test_convert_range_for_to_while_with_label() {
        check_assist(
            convert_range_for_to_while,
            "
fn foo() {
    'a: $0for mut i in 3..7 {
        foo(i);
    }
}
            ",
            "
fn foo() {
    let mut i = 3;
    'a: while i < 7 {
        foo(i);
        i += 1;
    }
}
            ",
        );
    }

    #[test]
    fn test_convert_range_for_to_while_step_by() {
        check_assist(
            convert_range_for_to_while,
            "
fn foo() {
    $0for mut i in (3..7).step_by(2) {
        foo(i);
    }
}
            ",
            "
fn foo() {
    let mut i = 3;
    while i < 7 {
        foo(i);
        i += 2;
    }
}
            ",
        );
    }

    #[test]
    fn test_convert_range_for_to_while_not_applicable_non_range() {
        check_assist_not_applicable(
            convert_range_for_to_while,
            "
fn foo() {
    let ident = 3..7;
    $0for mut i in ident {
        foo(i);
    }
}
            ",
        );
    }
}
