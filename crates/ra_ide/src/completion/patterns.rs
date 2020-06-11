use ra_syntax::{
    algo::non_trivia_sibling,
    ast::{self, LoopBodyOwner},
    match_ast, AstNode, Direction, NodeOrToken, SyntaxElement,
    SyntaxKind::*,
    SyntaxNode,
};

pub(crate) fn inside_impl(element: SyntaxElement) -> bool {
    let node = match element {
        NodeOrToken::Node(node) => node,
        NodeOrToken::Token(token) => token.parent(),
    };
    node.ancestors().find(|it| it.kind() == IMPL_DEF).is_some()
}

pub(crate) fn has_bind_pat_parent(element: SyntaxElement) -> bool {
    let node = match element {
        NodeOrToken::Node(node) => node,
        NodeOrToken::Token(token) => token.parent(),
    };
    node.ancestors().find(|it| it.kind() == BIND_PAT).is_some()
}

pub(crate) fn has_ref_pat_parent(element: SyntaxElement) -> bool {
    let node = match element {
        NodeOrToken::Node(node) => node,
        NodeOrToken::Token(token) => token.parent(),
    };
    node.ancestors().find(|it| it.kind() == REF_PAT).is_some()
}

pub(crate) fn goes_after_unsafe(element: SyntaxElement) -> bool {
    if let Some(token) = previous_non_triva_element(element).and_then(|it| it.into_token()) {
        if token.kind() == UNSAFE_KW {
            return true;
        }
    }
    false
}

pub(crate) fn has_block_expr_parent(element: SyntaxElement) -> bool {
    not_same_range_parent(element).filter(|it| it.kind() == BLOCK_EXPR).is_some()
}

pub(crate) fn has_item_list_parent(element: SyntaxElement) -> bool {
    not_same_range_parent(element).filter(|it| it.kind() == ITEM_LIST).is_some()
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

fn not_same_range_parent(element: SyntaxElement) -> Option<SyntaxNode> {
    let node = match element {
        NodeOrToken::Node(node) => node,
        NodeOrToken::Token(token) => token.parent(),
    };
    let range = node.text_range();
    node.ancestors().take_while(|it| it.text_range() == range).last().and_then(|it| it.parent())
}

fn previous_non_triva_element(element: SyntaxElement) -> Option<SyntaxElement> {
    // trying to get first non triva sibling if we have one
    let token_sibling = non_trivia_sibling(element.clone(), Direction::Prev);
    let mut wrapped = if let Some(sibling) = token_sibling {
        sibling
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
        non_trivia_sibling(NodeOrToken::Node(prev_sibling_node), Direction::Prev)?
    };
    //I think you can avoid this loop if you use SyntaxToken::prev_token -- unlike prev_sibling_or_token, it works across parents.
    // traversing the tree down to get the last token or node, i.e. the closest one
    loop {
        if let Some(token) = wrapped.as_token() {
            return Some(NodeOrToken::Token(token.clone()));
        } else {
            let new = wrapped.as_node().and_then(|n| n.last_child_or_token());
            if new.is_some() {
                wrapped = new.unwrap().clone();
            } else {
                return Some(wrapped);
            }
        }
    }
}
