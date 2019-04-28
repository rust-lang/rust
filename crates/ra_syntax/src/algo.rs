pub mod visit;

use itertools::Itertools;

use crate::{SyntaxNode, TextRange, TextUnit, AstNode, Direction, SyntaxToken, SyntaxElement};

pub use rowan::TokenAtOffset;

pub fn find_token_at_offset(node: &SyntaxNode, offset: TextUnit) -> TokenAtOffset<SyntaxToken> {
    match node.0.token_at_offset(offset) {
        TokenAtOffset::None => TokenAtOffset::None,
        TokenAtOffset::Single(n) => TokenAtOffset::Single(n.into()),
        TokenAtOffset::Between(l, r) => TokenAtOffset::Between(l.into(), r.into()),
    }
}

/// Returns ancestors of the node at the offset, sorted by length. This should
/// do the right thing at an edge, e.g. when searching for expressions at `{
/// <|>foo }` we will get the name reference instead of the whole block, which
/// we would get if we just did `find_token_at_offset(...).flat_map(|t|
/// t.parent().ancestors())`.
pub fn ancestors_at_offset(
    node: &SyntaxNode,
    offset: TextUnit,
) -> impl Iterator<Item = &SyntaxNode> {
    find_token_at_offset(node, offset)
        .map(|token| token.parent().ancestors())
        .kmerge_by(|node1, node2| node1.range().len() < node2.range().len())
}

/// Finds a node of specific Ast type at offset. Note that this is slightly
/// imprecise: if the cursor is strictly between two nodes of the desired type,
/// as in
///
/// ```no-run
/// struct Foo {}|struct Bar;
/// ```
///
/// then the shorter node will be silently preferred.
pub fn find_node_at_offset<N: AstNode>(syntax: &SyntaxNode, offset: TextUnit) -> Option<&N> {
    ancestors_at_offset(syntax, offset).find_map(N::cast)
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
