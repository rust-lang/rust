use ide_db::syntax_helpers::node_ext::is_pattern_cond;
use syntax::{
    T,
    ast::{self, AstNode},
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
    utils::invert_boolean_expression_legacy,
};

// Assist: invert_if
//
// This transforms if expressions of the form `if !x {A} else {B}` into `if x {B} else {A}`
// This also works with `!=`. This assist can only be applied with the cursor on `if`.
//
// ```
// fn main() {
//     if$0 !y { A } else { B }
// }
// ```
// ->
// ```
// fn main() {
//     if y { B } else { A }
// }
// ```
pub(crate) fn invert_if(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let if_keyword = ctx.find_token_syntax_at_offset(T![if])?;
    let expr = ast::IfExpr::cast(if_keyword.parent()?)?;
    let if_range = if_keyword.text_range();
    let cursor_in_range = if_range.contains_range(ctx.selection_trimmed());
    if !cursor_in_range {
        return None;
    }

    let cond = expr.condition()?;
    // This assist should not apply for if-let.
    if is_pattern_cond(cond.clone()) {
        return None;
    }

    let then_node = expr.then_branch()?.syntax().clone();
    let else_block = match expr.else_branch()? {
        ast::ElseBranch::Block(it) => it,
        ast::ElseBranch::IfExpr(_) => return None,
    };

    acc.add(AssistId::refactor_rewrite("invert_if"), "Invert if", if_range, |edit| {
        let flip_cond = invert_boolean_expression_legacy(cond.clone());
        edit.replace_ast(cond, flip_cond);

        let else_node = else_block.syntax();
        let else_range = else_node.text_range();
        let then_range = then_node.text_range();

        edit.replace(else_range, then_node.text());
        edit.replace(then_range, else_node.text());
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn invert_if_composite_condition() {
        check_assist(
            invert_if,
            "fn f() { i$0f x == 3 || x == 4 || x == 5 { 1 } else { 3 * 2 } }",
            "fn f() { if !(x == 3 || x == 4 || x == 5) { 3 * 2 } else { 1 } }",
        )
    }

    #[test]
    fn invert_if_remove_not_parentheses() {
        check_assist(
            invert_if,
            "fn f() { i$0f !(x == 3 || x == 4 || x == 5) { 3 * 2 } else { 1 } }",
            "fn f() { if x == 3 || x == 4 || x == 5 { 1 } else { 3 * 2 } }",
        )
    }

    #[test]
    fn invert_if_remove_inequality() {
        check_assist(
            invert_if,
            "fn f() { i$0f x != 3 { 1 } else { 3 + 2 } }",
            "fn f() { if x == 3 { 3 + 2 } else { 1 } }",
        )
    }

    #[test]
    fn invert_if_remove_not() {
        check_assist(
            invert_if,
            "fn f() { $0if !cond { 3 * 2 } else { 1 } }",
            "fn f() { if cond { 1 } else { 3 * 2 } }",
        )
    }

    #[test]
    fn invert_if_general_case() {
        check_assist(
            invert_if,
            "fn f() { i$0f cond { 3 * 2 } else { 1 } }",
            "fn f() { if !cond { 1 } else { 3 * 2 } }",
        )
    }

    #[test]
    fn invert_if_doesnt_apply_with_cursor_not_on_if() {
        check_assist_not_applicable(invert_if, "fn f() { if !$0cond { 3 * 2 } else { 1 } }")
    }

    #[test]
    fn invert_if_doesnt_apply_with_if_let() {
        check_assist_not_applicable(
            invert_if,
            "fn f() { i$0f let Some(_) = Some(1) { 1 } else { 0 } }",
        )
    }

    #[test]
    fn invert_if_option_case() {
        check_assist(
            invert_if,
            "fn f() { if$0 doc_style.is_some() { Class::DocComment } else { Class::Comment } }",
            "fn f() { if doc_style.is_none() { Class::Comment } else { Class::DocComment } }",
        )
    }

    #[test]
    fn invert_if_result_case() {
        check_assist(
            invert_if,
            "fn f() { i$0f doc_style.is_err() { Class::Err } else { Class::Ok } }",
            "fn f() { if doc_style.is_ok() { Class::Ok } else { Class::Err } }",
        )
    }
}
