pub mod visit;

use crate::{SyntaxNode, TextRange, TextUnit, AstNode, Direction, SyntaxToken, SyntaxElement};

pub use rowan::TokenAtOffset;

pub fn find_token_at_offset(node: &SyntaxNode, offset: TextUnit) -> TokenAtOffset<SyntaxToken> {
    match node.0.token_at_offset(offset) {
        TokenAtOffset::None => TokenAtOffset::None,
        TokenAtOffset::Single(n) => TokenAtOffset::Single(n.into()),
        TokenAtOffset::Between(l, r) => TokenAtOffset::Between(l.into(), r.into()),
    }
}

/// Finds a node of specific Ast type at offset. Note that this is slightly
/// imprecise: if the cursor is strictly between two nodes of the desired type,
/// as in
///
/// ```no-run
/// struct Foo {}|struct Bar;
/// ```
///
/// then the left node will be silently preferred.
pub fn find_node_at_offset<N: AstNode>(syntax: &SyntaxNode, offset: TextUnit) -> Option<&N> {
    find_token_at_offset(syntax, offset)
        .find_map(|leaf| leaf.parent().ancestors().find_map(N::cast))
}

/// Finds the first sibling in the given direction which is not `trivia`
pub fn non_trivia_sibling(element: SyntaxElement, direction: Direction) -> Option<SyntaxElement> {
    return match element {
        SyntaxElement::Node(node) => node.siblings_with_tokens(direction).skip(1).find(not_trivia),
        SyntaxElement::Token(token) => {
            token.siblings_with_tokens(direction).skip(1).find(not_trivia)
        }
    };

    fn not_trivia(element: &SyntaxElement) -> bool {
        match element {
            SyntaxElement::Node(_) => true,
            SyntaxElement::Token(token) => !token.kind().is_trivia(),
        }
    }
}

pub fn find_covering_element(root: &SyntaxNode, range: TextRange) -> SyntaxElement {
    root.0.covering_node(range).into()
}

// Replace with `std::iter::successors` in `1.34.0`
pub fn generate<T>(seed: Option<T>, step: impl Fn(&T) -> Option<T>) -> impl Iterator<Item = T> {
    ::itertools::unfold(seed, move |slot| {
        slot.take().map(|curr| {
            *slot = step(&curr);
            curr
        })
    })
}
