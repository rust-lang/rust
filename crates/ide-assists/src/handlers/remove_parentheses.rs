use syntax::{ast, AstNode, SyntaxKind, TextRange};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: remove_parentheses
//
// Removes useless parentheses.
//
// ```
// fn main() {
//     _ = $0(2) + 2;
// }
// ```
// ->
// ```
// fn main() {
//     _ = 2 + 2;
// }
// ```
pub(crate) fn remove_parentheses(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let parens = ctx.find_node_at_offset::<ast::ParenExpr>()?;
    let l_paren = parens.l_paren_token()?;
    let r_paren = parens.r_paren_token()?;

    let cursor_in_range = l_paren.text_range().contains_range(ctx.selection_trimmed())
        || r_paren.text_range().contains_range(ctx.selection_trimmed());
    if !cursor_in_range {
        return None;
    }

    // FIXME: check that precedence is right

    let delete_from_l = l_paren.text_range().start();
    let delete_to_l = match l_paren.next_token() {
        Some(it) if it.kind() == SyntaxKind::WHITESPACE => it.text_range().end(),
        _ => l_paren.text_range().end(),
    };

    let delete_from_r = match r_paren.prev_token() {
        Some(it) if it.kind() == SyntaxKind::WHITESPACE => it.text_range().start(),
        _ => r_paren.text_range().start(),
    };
    let delete_to_r = r_paren.text_range().end();

    let target = parens.syntax().text_range();
    acc.add(
        AssistId("remove_parentheses", AssistKind::Refactor),
        "Remove parentheses",
        target,
        |builder| {
            builder.delete(TextRange::new(delete_from_l, delete_to_l));
            builder.delete(TextRange::new(delete_from_r, delete_to_r));
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn remove_parens_simple() {
        check_assist(remove_parentheses, r#"fn f() { $0(2) + 2; }"#, r#"fn f() { 2 + 2; }"#);
        check_assist(remove_parentheses, r#"fn f() { ($02) + 2; }"#, r#"fn f() { 2 + 2; }"#);
        check_assist(remove_parentheses, r#"fn f() { (2)$0 + 2; }"#, r#"fn f() { 2 + 2; }"#);
        check_assist(remove_parentheses, r#"fn f() { (2$0) + 2; }"#, r#"fn f() { 2 + 2; }"#);
    }

    // We should not permit assist here and yet
    #[test]
    fn remove_parens_wrong() {
        check_assist(
            remove_parentheses,
            r#"fn f() { $0(2 + 2) * 8; }"#,
            r#"fn f() { 2 + 2 * 8; }"#,
        );
    }

    #[test]
    fn remove_parens_doesnt_apply_with_cursor_not_on_paren() {
        check_assist_not_applicable(remove_parentheses, r#"fn f() { (2 +$0 2) }"#);
        check_assist_not_applicable(remove_parentheses, r#"fn f() {$0 (2 + 2) }"#);
    }
}
