use syntax::{ast, AstNode};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: remove_parentheses
//
// Removes redundant parentheses.
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

    let cursor_in_range =
        parens.l_paren_token()?.text_range().contains_range(ctx.selection_trimmed())
            || parens.r_paren_token()?.text_range().contains_range(ctx.selection_trimmed());
    if !cursor_in_range {
        return None;
    }

    let expr = parens.expr()?;

    let parent = parens.syntax().parent()?;
    if expr.needs_parens_in(parent) {
        return None;
    }

    let target = parens.syntax().text_range();
    acc.add(
        AssistId("remove_parentheses", AssistKind::Refactor),
        "Remove redundant parentheses",
        target,
        |builder| builder.replace_ast(parens.into(), expr),
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

    #[test]
    fn remove_parens_closure() {
        check_assist(remove_parentheses, r#"fn f() { &$0(|| 42) }"#, r#"fn f() { &|| 42 }"#);

        check_assist_not_applicable(remove_parentheses, r#"fn f() { $0(|| 42).f() }"#);
    }

    #[test]
    fn remove_parens_if_let_chains() {
        check_assist_not_applicable(
            remove_parentheses,
            r#"fn f() { if let true = $0(true && true) {} }"#,
        );
    }

    #[test]
    fn remove_parens_associativity() {
        check_assist(
            remove_parentheses,
            r#"fn f() { $0(2 + 2) + 2; }"#,
            r#"fn f() { 2 + 2 + 2; }"#,
        );
        check_assist_not_applicable(remove_parentheses, r#"fn f() { 2 + $0(2 + 2); }"#);
    }

    #[test]
    fn remove_parens_precedence() {
        check_assist(
            remove_parentheses,
            r#"fn f() { $0(2 * 3) + 1; }"#,
            r#"fn f() { 2 * 3 + 1; }"#,
        );
        check_assist(remove_parentheses, r#"fn f() { ( $0(2) ); }"#, r#"fn f() { ( 2 ); }"#);
        check_assist(remove_parentheses, r#"fn f() { $0(2?)?; }"#, r#"fn f() { 2??; }"#);
        check_assist(remove_parentheses, r#"fn f() { f(($02 + 2)); }"#, r#"fn f() { f(2 + 2); }"#);
        check_assist(
            remove_parentheses,
            r#"fn f() { (1<2)&&$0(3>4); }"#,
            r#"fn f() { (1<2)&&3>4; }"#,
        );
    }

    #[test]
    fn remove_parens_doesnt_apply_precedence() {
        check_assist_not_applicable(remove_parentheses, r#"fn f() { $0(2 + 2) * 8; }"#);
        check_assist_not_applicable(remove_parentheses, r#"fn f() { $0(2 + 2).f(); }"#);
        check_assist_not_applicable(remove_parentheses, r#"fn f() { $0(2 + 2).await; }"#);
        check_assist_not_applicable(remove_parentheses, r#"fn f() { $0!(2..2); }"#);
    }

    #[test]
    fn remove_parens_doesnt_apply_with_cursor_not_on_paren() {
        check_assist_not_applicable(remove_parentheses, r#"fn f() { (2 +$0 2) }"#);
        check_assist_not_applicable(remove_parentheses, r#"fn f() {$0 (2 + 2) }"#);
    }

    #[test]
    fn remove_parens_doesnt_apply_when_expr_would_be_turned_into_a_statement() {
        check_assist_not_applicable(remove_parentheses, r#"fn x() -> u8 { $0({ 0 } + 1) }"#);
        check_assist_not_applicable(
            remove_parentheses,
            r#"fn x() -> u8 { $0(if true { 0 } else { 1 } + 1) }"#,
        );
        check_assist_not_applicable(remove_parentheses, r#"fn x() -> u8 { $0(loop {} + 1) }"#);
    }

    #[test]
    fn remove_parens_doesnt_apply_weird_syntax_and_edge_cases() {
        // removing `()` would break code because {} would be counted as the loop/if body
        check_assist_not_applicable(remove_parentheses, r#"fn f() { for _ in $0(0..{3}) {} }"#);
        check_assist_not_applicable(remove_parentheses, r#"fn f() { for _ in $0(S {}) {} }"#);
        check_assist_not_applicable(remove_parentheses, r#"fn f() { if $0(S {} == 2) {} }"#);
        check_assist_not_applicable(remove_parentheses, r#"fn f() { if $0(return) {} }"#);
    }

    #[test]
    fn remove_parens_return_with_value_followed_by_block() {
        check_assist(
            remove_parentheses,
            r#"fn f() { if $0(return ()) {} }"#,
            r#"fn f() { if return () {} }"#,
        );
    }

    #[test]
    fn remove_exprs_let_else_restrictions() {
        // `}` is not allowed before `else` here
        check_assist_not_applicable(
            remove_parentheses,
            r#"fn f() { let _ = $0(S{}) else { return }; }"#,
        );

        // logic operators can't directly appear in the let-else
        check_assist_not_applicable(
            remove_parentheses,
            r#"fn f() { let _ = $0(false || false) else { return }; }"#,
        );
        check_assist_not_applicable(
            remove_parentheses,
            r#"fn f() { let _ = $0(true && true) else { return }; }"#,
        );
    }

    #[test]
    fn remove_parens_weird_places() {
        check_assist(
            remove_parentheses,
            r#"fn f() { match () { _=>$0(()) } }"#,
            r#"fn f() { match () { _=>() } }"#,
        );

        check_assist(
            remove_parentheses,
            r#"fn x() -> u8 { { [$0({ 0 } + 1)] } }"#,
            r#"fn x() -> u8 { { [{ 0 } + 1] } }"#,
        );
    }

    #[test]
    fn remove_parens_return_dot_f() {
        check_assist(
            remove_parentheses,
            r#"fn f() { $0(return).f() }"#,
            r#"fn f() { return.f() }"#,
        );
    }

    #[test]
    fn remove_parens_prefix_then_return_something() {
        check_assist(
            remove_parentheses,
            r#"fn f() { &$0(return ()) }"#,
            r#"fn f() { &return () }"#,
        );
    }

    #[test]
    fn remove_parens_double_paren_stmt() {
        check_assist(
            remove_parentheses,
            r#"fn x() -> u8 { $0(({ 0 } + 1)) }"#,
            r#"fn x() -> u8 { ({ 0 } + 1) }"#,
        );

        check_assist(
            remove_parentheses,
            r#"fn x() -> u8 { (($0{ 0 } + 1)) }"#,
            r#"fn x() -> u8 { ({ 0 } + 1) }"#,
        );
    }

    #[test]
    fn remove_parens_im_tired_of_naming_tests() {
        check_assist(
            remove_parentheses,
            r#"fn f() { 2 + $0(return 2) }"#,
            r#"fn f() { 2 + return 2 }"#,
        );

        check_assist_not_applicable(remove_parentheses, r#"fn f() { $0(return 2) + 2 }"#);
    }
}
