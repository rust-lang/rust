//! FIXME: write short doc here

use std::{
    fmt,
    ops::{self, RangeInclusive},
};

use itertools::Itertools;
use rustc_hash::FxHashMap;
use text_edit::TextEditBuilder;

use crate::{
    AstNode, Direction, NodeOrToken, SyntaxElement, SyntaxKind, SyntaxNode, SyntaxNodePtr,
    SyntaxToken, TextRange, TextSize,
};

/// Returns ancestors of the node at the offset, sorted by length. This should
/// do the right thing at an edge, e.g. when searching for expressions at `{
/// <|>foo }` we will get the name reference instead of the whole block, which
/// we would get if we just did `find_token_at_offset(...).flat_map(|t|
/// t.parent().ancestors())`.
pub fn ancestors_at_offset(
    node: &SyntaxNode,
    offset: TextSize,
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
pub fn find_node_at_offset<N: AstNode>(syntax: &SyntaxNode, offset: TextSize) -> Option<N> {
    ancestors_at_offset(syntax, offset).find_map(N::cast)
}

pub fn find_node_at_range<N: AstNode>(syntax: &SyntaxNode, range: TextRange) -> Option<N> {
    find_covering_element(syntax, range).ancestors().find_map(N::cast)
}

/// Skip to next non `trivia` token
pub fn skip_trivia_token(mut token: SyntaxToken, direction: Direction) -> Option<SyntaxToken> {
    while token.kind().is_trivia() {
        token = match direction {
            Direction::Next => token.next_token()?,
            Direction::Prev => token.prev_token()?,
        }
    }
    Some(token)
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

pub fn least_common_ancestor(u: &SyntaxNode, v: &SyntaxNode) -> Option<SyntaxNode> {
    if u == v {
        return Some(u.clone());
    }

    let u_depth = u.ancestors().count();
    let v_depth = v.ancestors().count();
    let keep = u_depth.min(v_depth);

    let u_candidates = u.ancestors().skip(u_depth - keep);
    let v_canidates = v.ancestors().skip(v_depth - keep);
    let (res, _) = u_candidates.zip(v_canidates).find(|(x, y)| x == y)?;
    Some(res)
}

pub fn neighbor<T: AstNode>(me: &T, direction: Direction) -> Option<T> {
    me.syntax().siblings(direction).skip(1).find_map(T::cast)
}

pub fn has_errors(node: &SyntaxNode) -> bool {
    node.children().any(|it| it.kind() == SyntaxKind::ERROR)
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

    pub fn is_empty(&self) -> bool {
        self.replacements.is_empty()
    }
}

/// Finds minimal the diff, which, applied to `from`, will result in `to`.
///
/// Specifically, returns a map whose keys are descendants of `from` and values
/// are descendants of `to`, such that  `replace_descendants(from, map) == to`.
///
/// A trivial solution is a singleton map `{ from: to }`, but this function
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
        if lhs.kind() == rhs.kind()
            && lhs.text_range().len() == rhs.text_range().len()
            && match (&lhs, &rhs) {
                (NodeOrToken::Node(lhs), NodeOrToken::Node(rhs)) => {
                    lhs.green() == rhs.green() || lhs.text() == rhs.text()
                }
                (NodeOrToken::Token(lhs), NodeOrToken::Token(rhs)) => lhs.text() == rhs.text(),
                _ => false,
            }
        {
            return;
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
    to_insert: impl IntoIterator<Item = SyntaxElement>,
) -> SyntaxNode {
    let mut to_insert = to_insert.into_iter();
    _insert_children(parent, position, &mut to_insert)
}

fn _insert_children(
    parent: &SyntaxNode,
    position: InsertPosition<SyntaxElement>,
    to_insert: &mut dyn Iterator<Item = SyntaxElement>,
) -> SyntaxNode {
    let mut delta = TextSize::default();
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
    to_insert: impl IntoIterator<Item = SyntaxElement>,
) -> SyntaxNode {
    let mut to_insert = to_insert.into_iter();
    _replace_children(parent, to_delete, &mut to_insert)
}

fn _replace_children(
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

#[derive(Default)]
pub struct SyntaxRewriter<'a> {
    f: Option<Box<dyn Fn(&SyntaxElement) -> Option<SyntaxElement> + 'a>>,
    //FIXME: add debug_assertions that all elements are in fact from the same file.
    replacements: FxHashMap<SyntaxElement, Replacement>,
}

impl fmt::Debug for SyntaxRewriter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SyntaxRewriter").field("replacements", &self.replacements).finish()
    }
}

impl<'a> SyntaxRewriter<'a> {
    pub fn from_fn(f: impl Fn(&SyntaxElement) -> Option<SyntaxElement> + 'a) -> SyntaxRewriter<'a> {
        SyntaxRewriter { f: Some(Box::new(f)), replacements: FxHashMap::default() }
    }
    pub fn delete<T: Clone + Into<SyntaxElement>>(&mut self, what: &T) {
        let what = what.clone().into();
        let replacement = Replacement::Delete;
        self.replacements.insert(what, replacement);
    }
    pub fn replace<T: Clone + Into<SyntaxElement>>(&mut self, what: &T, with: &T) {
        let what = what.clone().into();
        let replacement = Replacement::Single(with.clone().into());
        self.replacements.insert(what, replacement);
    }
    pub fn replace_with_many<T: Clone + Into<SyntaxElement>>(
        &mut self,
        what: &T,
        with: Vec<SyntaxElement>,
    ) {
        let what = what.clone().into();
        let replacement = Replacement::Many(with);
        self.replacements.insert(what, replacement);
    }
    pub fn replace_ast<T: AstNode>(&mut self, what: &T, with: &T) {
        self.replace(what.syntax(), with.syntax())
    }

    pub fn rewrite(&self, node: &SyntaxNode) -> SyntaxNode {
        if self.f.is_none() && self.replacements.is_empty() {
            return node.clone();
        }
        self.rewrite_children(node)
    }

    pub fn rewrite_ast<N: AstNode>(self, node: &N) -> N {
        N::cast(self.rewrite(node.syntax())).unwrap()
    }

    /// Returns a node that encompasses all replacements to be done by this rewriter.
    ///
    /// Passing the returned node to `rewrite` will apply all replacements queued up in `self`.
    ///
    /// Returns `None` when there are no replacements.
    pub fn rewrite_root(&self) -> Option<SyntaxNode> {
        assert!(self.f.is_none());
        self.replacements
            .keys()
            .map(|element| match element {
                SyntaxElement::Node(it) => it.clone(),
                SyntaxElement::Token(it) => it.parent(),
            })
            // If we only have one replacement, we must return its parent node, since `rewrite` does
            // not replace the node passed to it.
            .map(|it| it.parent().unwrap_or(it))
            .fold1(|a, b| least_common_ancestor(&a, &b).unwrap())
    }

    fn replacement(&self, element: &SyntaxElement) -> Option<Replacement> {
        if let Some(f) = &self.f {
            assert!(self.replacements.is_empty());
            return f(element).map(Replacement::Single);
        }
        self.replacements.get(element).cloned()
    }

    fn rewrite_children(&self, node: &SyntaxNode) -> SyntaxNode {
        //  FIXME: this could be made much faster.
        let mut new_children = Vec::new();
        for child in node.children_with_tokens() {
            self.rewrite_self(&mut new_children, &child);
        }
        with_children(node, new_children)
    }

    fn rewrite_self(
        &self,
        acc: &mut Vec<NodeOrToken<rowan::GreenNode, rowan::GreenToken>>,
        element: &SyntaxElement,
    ) {
        if let Some(replacement) = self.replacement(&element) {
            match replacement {
                Replacement::Single(NodeOrToken::Node(it)) => {
                    acc.push(NodeOrToken::Node(it.green().clone()))
                }
                Replacement::Single(NodeOrToken::Token(it)) => {
                    acc.push(NodeOrToken::Token(it.green().clone()))
                }
                Replacement::Many(replacements) => {
                    acc.extend(replacements.iter().map(|it| match it {
                        NodeOrToken::Node(it) => NodeOrToken::Node(it.green().clone()),
                        NodeOrToken::Token(it) => NodeOrToken::Token(it.green().clone()),
                    }))
                }
                Replacement::Delete => (),
            };
            return;
        }
        let res = match element {
            NodeOrToken::Token(it) => NodeOrToken::Token(it.green().clone()),
            NodeOrToken::Node(it) => NodeOrToken::Node(self.rewrite_children(it).green().clone()),
        };
        acc.push(res)
    }
}

impl ops::AddAssign for SyntaxRewriter<'_> {
    fn add_assign(&mut self, rhs: SyntaxRewriter) {
        assert!(rhs.f.is_none());
        self.replacements.extend(rhs.replacements)
    }
}

#[derive(Clone, Debug)]
enum Replacement {
    Delete,
    Single(SyntaxElement),
    Many(Vec<SyntaxElement>),
}

fn with_children(
    parent: &SyntaxNode,
    new_children: Vec<NodeOrToken<rowan::GreenNode, rowan::GreenToken>>,
) -> SyntaxNode {
    let len = new_children.iter().map(|it| it.text_len()).sum::<TextSize>();
    let new_node = rowan::GreenNode::new(rowan::SyntaxKind(parent.kind() as u16), new_children);
    let new_root_node = parent.replace_with(new_node);
    let new_root_node = SyntaxNode::new_root(new_root_node);

    // FIXME: use a more elegant way to re-fetch the node (#1185), make
    // `range` private afterwards
    let mut ptr = SyntaxNodePtr::new(parent);
    ptr.range = TextRange::at(ptr.range.start(), len);
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
