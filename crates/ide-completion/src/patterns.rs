//! Patterns telling us certain facts about current syntax element, they are used in completion context
//!
//! Most logic in this module first expands the token below the cursor to a maximum node that acts similar to the token itself.
//! This means we for example expand a NameRef token to its outermost Path node, as semantically these act in the same location
//! and the completions usually query for path specific things on the Path context instead. This simplifies some location handling.

use syntax::{
    ast::{self, HasLoopBody},
    match_ast, AstNode, SyntaxElement,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken,
};

#[cfg(test)]
use crate::tests::check_pattern_is_applicable;

pub(crate) fn previous_token(element: SyntaxElement) -> Option<SyntaxToken> {
    element.into_token().and_then(previous_non_trivia_token)
}

pub(crate) fn is_in_token_of_for_loop(element: SyntaxElement) -> bool {
    // oh my ...
    (|| {
        let syntax_token = element.into_token()?;
        let range = syntax_token.text_range();
        let for_expr = syntax_token.parent_ancestors().find_map(ast::ForExpr::cast)?;

        // check if the current token is the `in` token of a for loop
        if let Some(token) = for_expr.in_token() {
            return Some(syntax_token == token);
        }
        let pat = for_expr.pat()?;
        if range.end() < pat.syntax().text_range().end() {
            // if we are inside or before the pattern we can't be at the `in` token position
            return None;
        }
        let next_sibl = next_non_trivia_sibling(pat.syntax().clone().into())?;
        Some(match next_sibl {
            // the loop body is some node, if our token is at the start we are at the `in` position,
            // otherwise we could be in a recovered expression, we don't wanna ruin completions there
            syntax::NodeOrToken::Node(n) => n.text_range().start() == range.start(),
            // the loop body consists of a single token, if we are this we are certainly at the `in` token position
            syntax::NodeOrToken::Token(t) => t == syntax_token,
        })
    })()
    .unwrap_or(false)
}

#[test]
fn test_for_is_prev2() {
    check_pattern_is_applicable(r"fn __() { for i i$0 }", is_in_token_of_for_loop);
}

pub(crate) fn is_in_loop_body(node: &SyntaxNode) -> bool {
    node.ancestors()
        .take_while(|it| it.kind() != FN && it.kind() != CLOSURE_EXPR)
        .find_map(|it| {
            let loop_body = match_ast! {
                match it {
                    ast::ForExpr(it) => it.loop_body(),
                    ast::WhileExpr(it) => it.loop_body(),
                    ast::LoopExpr(it) => it.loop_body(),
                    _ => None,
                }
            };
            loop_body.filter(|it| it.syntax().text_range().contains_range(node.text_range()))
        })
        .is_some()
}

fn previous_non_trivia_token(token: SyntaxToken) -> Option<SyntaxToken> {
    let mut token = token.prev_token();
    while let Some(inner) = token {
        if !inner.kind().is_trivia() {
            return Some(inner);
        } else {
            token = inner.prev_token();
        }
    }
    None
}

fn next_non_trivia_sibling(ele: SyntaxElement) -> Option<SyntaxElement> {
    let mut e = ele.next_sibling_or_token();
    while let Some(inner) = e {
        if !inner.kind().is_trivia() {
            return Some(inner);
        } else {
            e = inner.next_sibling_or_token();
        }
    }
    None
}
