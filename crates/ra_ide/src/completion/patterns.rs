use ra_syntax::{
    algo::non_trivia_sibling,
    ast::{self, LoopBodyOwner},
    match_ast, AstNode, Direction, NodeOrToken, SyntaxElement,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken,
};

pub(crate) fn inside_impl(element: SyntaxElement) -> bool {
    element.ancestors().find(|it| it.kind() == IMPL_DEF).is_some()
}

pub(crate) fn inside_trait(element: SyntaxElement) -> bool {
    element.ancestors().find(|it| it.kind() == TRAIT_DEF).is_some()
}

pub(crate) fn has_bind_pat_parent(element: SyntaxElement) -> bool {
    element.ancestors().find(|it| it.kind() == BIND_PAT).is_some()
}

pub(crate) fn has_ref_pat_parent(element: SyntaxElement) -> bool {
    element.ancestors().find(|it| it.kind() == REF_PAT).is_some()
}

pub(crate) fn goes_after_unsafe(element: SyntaxElement) -> bool {
    element
        .into_token()
        .and_then(|it| previous_non_trivia_token(it))
        .filter(|it| it.kind() == UNSAFE_KW)
        .is_some()
}

pub(crate) fn has_block_expr_parent(element: SyntaxElement) -> bool {
    not_same_range_ancestor(element).filter(|it| it.kind() == BLOCK_EXPR).is_some()
}

pub(crate) fn has_item_list_parent(element: SyntaxElement) -> bool {
    not_same_range_ancestor(element).filter(|it| it.kind() == ITEM_LIST).is_some()
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
