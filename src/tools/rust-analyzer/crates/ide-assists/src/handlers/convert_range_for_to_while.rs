use ide_db::assists::AssistId;
use itertools::Itertools;
use syntax::{
    AstNode, SyntaxElement,
    SyntaxKind::WHITESPACE,
    T,
    algo::previous_non_trivia_token,
    ast::{
        self, HasArgList, HasLoopBody, HasName, RangeItem, edit::AstNodeEdit, make,
        syntax_factory::SyntaxFactory,
    },
    syntax_editor::{Element, Position, SyntaxEditor},
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
    let body = for_.loop_body()?.stmt_list()?;
    let label = for_.label();

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
            process_loop_body(
                body,
                label,
                &mut edit,
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

fn process_loop_body(
    body: ast::StmtList,
    label: Option<ast::Label>,
    edit: &mut SyntaxEditor,
    incrementer: Vec<SyntaxElement>,
) -> Option<()> {
    let last = previous_non_trivia_token(body.r_curly_token()?)?.syntax_element();

    let new_body = body.indent(1.into()).clone_subtree();
    let mut continues = vec![];
    collect_continue_to(
        &mut continues,
        &label.and_then(|it| it.lifetime()),
        new_body.syntax(),
        false,
    );

    if continues.is_empty() {
        edit.insert_all(Position::after(last), incrementer);
        return Some(());
    }

    let mut children = body
        .syntax()
        .children_with_tokens()
        .filter(|it| !matches!(it.kind(), WHITESPACE | T!['{'] | T!['}']));
    let first = children.next()?;
    let block_content = first.clone()..=children.last().unwrap_or(first);

    let continue_label = make::lifetime("'cont");
    let break_expr = make::expr_break(Some(continue_label.clone()), None).clone_for_update();
    let mut new_edit = SyntaxEditor::new(new_body.syntax().clone());
    for continue_expr in &continues {
        new_edit.replace(continue_expr.syntax(), break_expr.syntax());
    }
    let new_body = new_edit.finish().new_root().clone();
    let elements = itertools::chain(
        [
            continue_label.syntax().clone_for_update().syntax_element(),
            make::token(T![:]).syntax_element(),
            make::tokens::single_space().syntax_element(),
            new_body.syntax_element(),
        ],
        incrementer,
    );
    edit.replace_all(block_content, elements.collect());

    Some(())
}

fn collect_continue_to(
    acc: &mut Vec<ast::ContinueExpr>,
    label: &Option<ast::Lifetime>,
    node: &syntax::SyntaxNode,
    only_label: bool,
) {
    let match_label = |it: &Option<ast::Lifetime>, label: &Option<ast::Lifetime>| match (it, label)
    {
        (None, _) => !only_label,
        (Some(a), Some(b)) if a.text() == b.text() => true,
        _ => false,
    };
    if let Some(expr) = ast::ContinueExpr::cast(node.clone())
        && match_label(&expr.lifetime(), label)
    {
        acc.push(expr);
    } else if let Some(any_loop) = ast::AnyHasLoopBody::cast(node.clone()) {
        if match_label(label, &any_loop.label().and_then(|it| it.lifetime())) {
            return;
        }
        for children in node.children() {
            collect_continue_to(acc, label, &children, true);
        }
    } else {
        for children in node.children() {
            collect_continue_to(acc, label, &children, only_label);
        }
    }
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
    fn test_convert_range_for_to_while_with_continue() {
        check_assist(
            convert_range_for_to_while,
            "
fn foo() {
    $0for mut i in 3..7 {
        foo(i);
        continue;
        loop { break; continue }
        bar(i);
    }
}
            ",
            "
fn foo() {
    let mut i = 3;
    while i < 7 {
        'cont: {
            foo(i);
            break 'cont;
            loop { break; continue }
            bar(i);
        }
        i += 1;
    }
}
            ",
        );

        check_assist(
            convert_range_for_to_while,
            "
fn foo() {
    'x: $0for mut i in 3..7 {
        foo(i);
        continue 'x;
        loop { break; continue 'x }
        'x: loop { continue 'x }
        bar(i);
    }
}
            ",
            "
fn foo() {
    let mut i = 3;
    'x: while i < 7 {
        'cont: {
            foo(i);
            break 'cont;
            loop { break; break 'cont }
            'x: loop { continue 'x }
            bar(i);
        }
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
