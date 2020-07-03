use ra_syntax::{
    ast::{self, AstNode},
    T,
};

use crate::{
    assist_context::{AssistContext, Assists},
    utils::invert_boolean_expression,
    AssistId, AssistKind,
};

// Assist: invert_if
//
// Apply invert_if
// This transforms if expressions of the form `if !x {A} else {B}` into `if x {B} else {A}`
// This also works with `!=`. This assist can only be applied with the cursor
// on `if`.
//
// ```
// fn main() {
//     if<|> !y { A } else { B }
// }
// ```
// ->
// ```
// fn main() {
//     if y { B } else { A }
// }
// ```

pub(crate) fn invert_if(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let if_keyword = ctx.find_token_at_offset(T![if])?;
    let expr = ast::IfExpr::cast(if_keyword.parent())?;
    let if_range = if_keyword.text_range();
    let cursor_in_range = if_range.contains_range(ctx.frange.range);
    if !cursor_in_range {
        return None;
    }

    // This assist should not apply for if-let.
    if expr.condition()?.pat().is_some() {
        return None;
    }

    let cond = expr.condition()?.expr()?;
    let then_node = expr.then_branch()?.syntax().clone();
    let else_block = match expr.else_branch()? {
        ast::ElseBranch::Block(it) => it,
        ast::ElseBranch::IfExpr(_) => return None,
    };

    let cond_range = cond.syntax().text_range();
    let flip_cond = invert_boolean_expression(cond);
    let else_node = else_block.syntax();
    let else_range = else_node.text_range();
    let then_range = then_node.text_range();
    acc.add(AssistId("invert_if", AssistKind::RefactorRewrite), "Invert if", if_range, |edit| {
        edit.replace(cond_range, flip_cond.syntax().text());
        edit.replace(else_range, then_node.text());
        edit.replace(then_range, else_node.text());
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn invert_if_remove_inequality() {
        check_assist(
            invert_if,
            "fn f() { i<|>f x != 3 { 1 } else { 3 + 2 } }",
            "fn f() { if x == 3 { 3 + 2 } else { 1 } }",
        )
    }

    #[test]
    fn invert_if_remove_not() {
        check_assist(
            invert_if,
            "fn f() { <|>if !cond { 3 * 2 } else { 1 } }",
            "fn f() { if cond { 1 } else { 3 * 2 } }",
        )
    }

    #[test]
    fn invert_if_general_case() {
        check_assist(
            invert_if,
            "fn f() { i<|>f cond { 3 * 2 } else { 1 } }",
            "fn f() { if !cond { 1 } else { 3 * 2 } }",
        )
    }

    #[test]
    fn invert_if_doesnt_apply_with_cursor_not_on_if() {
        check_assist_not_applicable(invert_if, "fn f() { if !<|>cond { 3 * 2 } else { 1 } }")
    }

    #[test]
    fn invert_if_doesnt_apply_with_if_let() {
        check_assist_not_applicable(
            invert_if,
            "fn f() { i<|>f let Some(_) = Some(1) { 1 } else { 0 } }",
        )
    }
}
