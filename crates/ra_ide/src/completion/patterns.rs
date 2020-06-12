//! Patterns telling us certain facts about current syntax element, they are used in completion context

use ra_syntax::{
    algo::non_trivia_sibling,
    ast::{self, LoopBodyOwner},
    match_ast, AstNode, Direction, NodeOrToken, SyntaxElement,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken,
};

pub(crate) fn has_trait_parent(element: SyntaxElement) -> bool {
    not_same_range_ancestor(element)
        .filter(|it| it.kind() == ITEM_LIST)
        .and_then(|it| it.parent())
        .filter(|it| it.kind() == TRAIT_DEF)
        .is_some()
}

pub(crate) fn has_impl_parent(element: SyntaxElement) -> bool {
    not_same_range_ancestor(element)
        .filter(|it| it.kind() == ITEM_LIST)
        .and_then(|it| it.parent())
        .filter(|it| it.kind() == IMPL_DEF)
        .is_some()
}

pub(crate) fn has_block_expr_parent(element: SyntaxElement) -> bool {
    not_same_range_ancestor(element).filter(|it| it.kind() == BLOCK_EXPR).is_some()
}

pub(crate) fn has_bind_pat_parent(element: SyntaxElement) -> bool {
    element.ancestors().find(|it| it.kind() == BIND_PAT).is_some()
}

pub(crate) fn has_ref_parent(element: SyntaxElement) -> bool {
    not_same_range_ancestor(element)
        .filter(|it| it.kind() == REF_PAT || it.kind() == REF_EXPR)
        .is_some()
}

pub(crate) fn is_match_arm(element: SyntaxElement) -> bool {
    not_same_range_ancestor(element.clone()).filter(|it| it.kind() == MATCH_ARM).is_some()
        && previous_sibling_or_ancestor_sibling(element)
            .and_then(|it| it.into_token())
            .filter(|it| it.kind() == FAT_ARROW)
            .is_some()
}

pub(crate) fn unsafe_is_prev(element: SyntaxElement) -> bool {
    element
        .into_token()
        .and_then(|it| previous_non_trivia_token(it))
        .filter(|it| it.kind() == UNSAFE_KW)
        .is_some()
}

pub(crate) fn if_is_prev(element: SyntaxElement) -> bool {
    element
        .into_token()
        .and_then(|it| previous_non_trivia_token(it))
        .filter(|it| it.kind() == IF_KW)
        .is_some()
}

pub(crate) fn has_trait_as_prev_sibling(element: SyntaxElement) -> bool {
    previous_sibling_or_ancestor_sibling(element).filter(|it| it.kind() == TRAIT_DEF).is_some()
}

pub(crate) fn has_impl_as_prev_sibling(element: SyntaxElement) -> bool {
    previous_sibling_or_ancestor_sibling(element).filter(|it| it.kind() == IMPL_DEF).is_some()
}

pub(crate) fn is_in_loop_body(element: SyntaxElement) -> bool {
    let leaf = match element {
        NodeOrToken::Node(node) => node,
        NodeOrToken::Token(token) => token.parent(),
    };
    for node in leaf.ancestors() {
        if node.kind() == FN_DEF || node.kind() == LAMBDA_EXPR {
            break;
        }
        let loop_body = match_ast! {
            match node {
                ast::ForExpr(it) => it.loop_body(),
                ast::WhileExpr(it) => it.loop_body(),
                ast::LoopExpr(it) => it.loop_body(),
                _ => None,
            }
        };
        if let Some(body) = loop_body {
            if body.syntax().text_range().contains_range(leaf.text_range()) {
                return true;
            }
        }
    }
    false
}

fn not_same_range_ancestor(element: SyntaxElement) -> Option<SyntaxNode> {
    element
        .ancestors()
        .take_while(|it| it.text_range() == element.text_range())
        .last()
        .and_then(|it| it.parent())
}

fn previous_non_trivia_token(token: SyntaxToken) -> Option<SyntaxToken> {
    let mut token = token.prev_token();
    while let Some(inner) = token.clone() {
        if !inner.kind().is_trivia() {
            return Some(inner);
        } else {
            token = inner.prev_token();
        }
    }
    None
}

fn previous_sibling_or_ancestor_sibling(element: SyntaxElement) -> Option<SyntaxElement> {
    let token_sibling = non_trivia_sibling(element.clone(), Direction::Prev);
    if let Some(sibling) = token_sibling {
        Some(sibling)
    } else {
        // if not trying to find first ancestor which has such a sibling
        let node = match element {
            NodeOrToken::Node(node) => node,
            NodeOrToken::Token(token) => token.parent(),
        };
        let range = node.text_range();
        let top_node = node.ancestors().take_while(|it| it.text_range() == range).last()?;
        let prev_sibling_node = top_node.ancestors().find(|it| {
            non_trivia_sibling(NodeOrToken::Node(it.to_owned()), Direction::Prev).is_some()
        })?;
        non_trivia_sibling(NodeOrToken::Node(prev_sibling_node), Direction::Prev)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        has_bind_pat_parent, has_block_expr_parent, has_impl_as_prev_sibling, has_impl_parent,
        has_ref_parent, has_trait_as_prev_sibling, has_trait_parent, if_is_prev, is_match_arm,
        unsafe_is_prev,
    };
    use crate::completion::test_utils::check_pattern_is_applicable;

    #[test]
    fn test_unsafe_is_prev() {
        check_pattern_is_applicable(r"unsafe i<|>", unsafe_is_prev);
    }

    #[test]
    fn test_if_is_prev() {
        check_pattern_is_applicable(r"if l<|>", if_is_prev);
    }

    #[test]
    fn test_has_trait_parent() {
        check_pattern_is_applicable(r"trait A { f<|> }", has_trait_parent);
    }

    #[test]
    fn test_has_impl_parent() {
        check_pattern_is_applicable(r"impl A { f<|> }", has_impl_parent);
    }

    #[test]
    fn test_has_trait_as_prev_sibling() {
        check_pattern_is_applicable(r"trait A w<|> {}", has_trait_as_prev_sibling);
    }

    #[test]
    fn test_has_impl_as_prev_sibling() {
        check_pattern_is_applicable(r"impl A w<|> {}", has_impl_as_prev_sibling);
    }

    #[test]
    fn test_parent_block_expr() {
        check_pattern_is_applicable(r"fn my_fn() { let a = 2; f<|> }", has_block_expr_parent);
    }

    #[test]
    fn test_has_ref_pat_parent_in_func_parameters() {
        check_pattern_is_applicable(r"fn my_fn(&m<|>) {}", has_ref_parent);
    }

    #[test]
    fn test_has_ref_pat_parent_in_let_statement() {
        check_pattern_is_applicable(r"fn my() { let &m<|> }", has_ref_parent);
    }

    #[test]
    fn test_has_bind_pat_parent_in_func_parameters() {
        check_pattern_is_applicable(r"fn my_fn(m<|>) {}", has_bind_pat_parent);
    }

    #[test]
    fn test_has_bind_pat_parent_in_let_statement() {
        check_pattern_is_applicable(r"fn my_fn() { let m<|> }", has_bind_pat_parent);
    }

    #[test]
    fn test_is_match_arm() {
        check_pattern_is_applicable(r"fn my_fn() { match () { () => m<|> } }", is_match_arm);
    }
}
