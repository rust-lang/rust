pub mod visit;

use rowan::TransparentNewType;

use crate::{SyntaxNode, TextRange, TextUnit, AstNode, Direction};

pub use rowan::LeafAtOffset;

pub fn find_leaf_at_offset(node: &SyntaxNode, offset: TextUnit) -> LeafAtOffset<&SyntaxNode> {
    match node.0.leaf_at_offset(offset) {
        LeafAtOffset::None => LeafAtOffset::None,
        LeafAtOffset::Single(n) => LeafAtOffset::Single(SyntaxNode::from_repr(n)),
        LeafAtOffset::Between(l, r) => {
            LeafAtOffset::Between(SyntaxNode::from_repr(l), SyntaxNode::from_repr(r))
        }
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
    find_leaf_at_offset(syntax, offset).find_map(|leaf| leaf.ancestors().find_map(N::cast))
}

/// Finds the first sibling in the given direction which is not `trivia`
pub fn non_trivia_sibling(node: &SyntaxNode, direction: Direction) -> Option<&SyntaxNode> {
    node.siblings(direction).skip(1).find(|node| !node.kind().is_trivia())
}

pub fn find_covering_node(root: &SyntaxNode, range: TextRange) -> &SyntaxNode {
    SyntaxNode::from_repr(root.0.covering_node(range))
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
