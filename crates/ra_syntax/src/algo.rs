//! FIXME: write short doc here

use std::ops::RangeInclusive;

use itertools::Itertools;
use ra_text_edit::TextEditBuilder;
use rustc_hash::FxHashMap;

use crate::{
    AstNode, Direction, NodeOrToken, SyntaxElement, SyntaxNode, SyntaxNodePtr, TextRange, TextUnit,
};

/// Returns ancestors of the node at the offset, sorted by length. This should
/// do the right thing at an edge, e.g. when searching for expressions at `{
/// <|>foo }` we will get the name reference instead of the whole block, which
/// we would get if we just did `find_token_at_offset(...).flat_map(|t|
/// t.parent().ancestors())`.
pub fn ancestors_at_offset(
    node: &SyntaxNode,
    offset: TextUnit,
) -> impl Iterator<Item = SyntaxNode> {
    node.token_at_offset(offset)
        .map(|token| token.parent().ancestors())
        .kmerge_by(|node1, node2| node1.text_range().len() < node2.text_range().len())
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
pub fn find_node_at_offset<N: AstNode>(syntax: &SyntaxNode, offset: TextUnit) -> Option<N> {
    ancestors_at_offset(syntax, offset).find_map(N::cast)
}

/// Finds the first sibling in the given direction which is not `trivia`
pub fn non_trivia_sibling(element: SyntaxElement, direction: Direction) -> Option<SyntaxElement> {
    return match element {
        NodeOrToken::Node(node) => node.siblings_with_tokens(direction).skip(1).find(not_trivia),
        NodeOrToken::Token(token) => token.siblings_with_tokens(direction).skip(1).find(not_trivia),
    };

    fn not_trivia(element: &SyntaxElement) -> bool {
        match element {
            NodeOrToken::Node(_) => true,
            NodeOrToken::Token(token) => !token.kind().is_trivia(),
        }
    }
}

pub fn find_covering_element(root: &SyntaxNode, range: TextRange) -> SyntaxElement {
    root.covering_element(range)
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum InsertPosition<T> {
    First,
    Last,
    Before(T),
    After(T),
}

pub struct TreeDiff {
    replacements: FxHashMap<SyntaxElement, SyntaxElement>,
}

impl TreeDiff {
    pub fn into_text_edit(&self, builder: &mut TextEditBuilder) {
        for (from, to) in self.replacements.iter() {
            builder.replace(from.text_range(), to.to_string())
        }
    }
}

/// Finds minimal the diff, which, applied to `from`, will result in `to`.
///
/// Specifically, returns a map whose keys are descendants of `from` and values
/// are descendants of `to`, such that  `replace_descendants(from, map) == to`.
///
/// A trivial solution is a singletom map `{ from: to }`, but this function
/// tries to find a more fine-grained diff.
pub fn diff(from: &SyntaxNode, to: &SyntaxNode) -> TreeDiff {
    let mut buf = FxHashMap::default();
    // FIXME: this is both horrible inefficient and gives larger than
    // necessary diff. I bet there's a cool algorithm to diff trees properly.
    go(&mut buf, from.clone().into(), to.clone().into());
    return TreeDiff { replacements: buf };

    fn go(
        buf: &mut FxHashMap<SyntaxElement, SyntaxElement>,
        lhs: SyntaxElement,
        rhs: SyntaxElement,
    ) {
        if lhs.kind() == rhs.kind() && lhs.text_range().len() == rhs.text_range().len() {
            if match (&lhs, &rhs) {
                (NodeOrToken::Node(lhs), NodeOrToken::Node(rhs)) => {
                    lhs.green() == rhs.green() || lhs.text() == rhs.text()
                }
                (NodeOrToken::Token(lhs), NodeOrToken::Token(rhs)) => lhs.text() == rhs.text(),
                _ => false,
            } {
                return;
            }
        }
        if let (Some(lhs), Some(rhs)) = (lhs.as_node(), rhs.as_node()) {
            if lhs.children_with_tokens().count() == rhs.children_with_tokens().count() {
                for (lhs, rhs) in lhs.children_with_tokens().zip(rhs.children_with_tokens()) {
                    go(buf, lhs, rhs)
                }
                return;
            }
        }
        buf.insert(lhs, rhs);
    }
}

/// Adds specified children (tokens or nodes) to the current node at the
/// specific position.
///
/// This is a type-unsafe low-level editing API, if you need to use it,
/// prefer to create a type-safe abstraction on top of it instead.
pub fn insert_children(
    parent: &SyntaxNode,
    position: InsertPosition<SyntaxElement>,
    to_insert: &mut dyn Iterator<Item = SyntaxElement>,
) -> SyntaxNode {
    let mut delta = TextUnit::default();
    let to_insert = to_insert.map(|element| {
        delta += element.text_range().len();
        to_green_element(element)
    });

    let mut old_children = parent.green().children().map(|it| match it {
        NodeOrToken::Token(it) => NodeOrToken::Token(it.clone()),
        NodeOrToken::Node(it) => NodeOrToken::Node(it.clone()),
    });

    let new_children = match &position {
        InsertPosition::First => to_insert.chain(old_children).collect::<Vec<_>>(),
        InsertPosition::Last => old_children.chain(to_insert).collect::<Vec<_>>(),
        InsertPosition::Before(anchor) | InsertPosition::After(anchor) => {
            let take_anchor = if let InsertPosition::After(_) = position { 1 } else { 0 };
            let split_at = position_of_child(parent, anchor.clone()) + take_anchor;
            let before = old_children.by_ref().take(split_at).collect::<Vec<_>>();
            before.into_iter().chain(to_insert).chain(old_children).collect::<Vec<_>>()
        }
    };

    with_children(parent, new_children)
}

/// Replaces all nodes in `to_delete` with nodes from `to_insert`
///
/// This is a type-unsafe low-level editing API, if you need to use it,
/// prefer to create a type-safe abstraction on top of it instead.
pub fn replace_children(
    parent: &SyntaxNode,
    to_delete: RangeInclusive<SyntaxElement>,
    to_insert: &mut dyn Iterator<Item = SyntaxElement>,
) -> SyntaxNode {
    let start = position_of_child(parent, to_delete.start().clone());
    let end = position_of_child(parent, to_delete.end().clone());
    let mut old_children = parent.green().children().map(|it| match it {
        NodeOrToken::Token(it) => NodeOrToken::Token(it.clone()),
        NodeOrToken::Node(it) => NodeOrToken::Node(it.clone()),
    });

    let before = old_children.by_ref().take(start).collect::<Vec<_>>();
    let new_children = before
        .into_iter()
        .chain(to_insert.map(to_green_element))
        .chain(old_children.skip(end + 1 - start))
        .collect::<Vec<_>>();
    with_children(parent, new_children)
}

/// Replaces descendants in the node, according to the mapping.
///
/// This is a type-unsafe low-level editing API, if you need to use it, prefer
/// to create a type-safe abstraction on top of it instead.
pub fn replace_descendants(
    parent: &SyntaxNode,
    map: &FxHashMap<SyntaxElement, SyntaxElement>,
) -> SyntaxNode {
    //  FIXME: this could be made much faster.
    let new_children = parent.children_with_tokens().map(|it| go(map, it)).collect::<Vec<_>>();
    return with_children(parent, new_children);

    fn go(
        map: &FxHashMap<SyntaxElement, SyntaxElement>,
        element: SyntaxElement,
    ) -> NodeOrToken<rowan::GreenNode, rowan::GreenToken> {
        if let Some(replacement) = map.get(&element) {
            return match replacement {
                NodeOrToken::Node(it) => NodeOrToken::Node(it.green().clone()),
                NodeOrToken::Token(it) => NodeOrToken::Token(it.green().clone()),
            };
        }
        match element {
            NodeOrToken::Token(it) => NodeOrToken::Token(it.green().clone()),
            NodeOrToken::Node(it) => {
                NodeOrToken::Node(replace_descendants(&it, map).green().clone())
            }
        }
    }
}

fn with_children(
    parent: &SyntaxNode,
    new_children: Vec<NodeOrToken<rowan::GreenNode, rowan::GreenToken>>,
) -> SyntaxNode {
    let len = new_children.iter().map(|it| it.text_len()).sum::<TextUnit>();
    let new_node = rowan::GreenNode::new(rowan::SyntaxKind(parent.kind() as u16), new_children);
    let new_root_node = parent.replace_with(new_node);
    let new_root_node = SyntaxNode::new_root(new_root_node);

    // FIXME: use a more elegant way to re-fetch the node (#1185), make
    // `range` private afterwards
    let mut ptr = SyntaxNodePtr::new(parent);
    ptr.range = TextRange::offset_len(ptr.range().start(), len);
    ptr.to_node(&new_root_node)
}

fn position_of_child(parent: &SyntaxNode, child: SyntaxElement) -> usize {
    parent
        .children_with_tokens()
        .position(|it| it == child)
        .expect("element is not a child of current element")
}

fn to_green_element(element: SyntaxElement) -> NodeOrToken<rowan::GreenNode, rowan::GreenToken> {
    match element {
        NodeOrToken::Node(it) => it.green().clone().into(),
        NodeOrToken::Token(it) => it.green().clone().into(),
    }
}
