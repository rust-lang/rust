pub mod visit;

use rowan::TransparentNewType;

use crate::{SyntaxNode, TextRange, TextUnit};

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

pub fn find_covering_node(root: &SyntaxNode, range: TextRange) -> &SyntaxNode {
    SyntaxNode::from_repr(root.0.covering_node(range))
}

pub fn generate<T>(seed: Option<T>, step: impl Fn(&T) -> Option<T>) -> impl Iterator<Item = T> {
    ::itertools::unfold(seed, move |slot| {
        slot.take().map(|curr| {
            *slot = step(&curr);
            curr
        })
    })
}
