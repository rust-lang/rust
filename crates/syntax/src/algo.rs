//! FIXME: write short doc here

use std::{
    fmt,
    hash::BuildHasherDefault,
    ops::{self, RangeInclusive},
};

use indexmap::IndexMap;
use itertools::Itertools;
use rustc_hash::FxHashMap;
use test_utils::mark;
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
/// ```no_run
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

type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<rustc_hash::FxHasher>>;

#[derive(Debug)]
pub struct TreeDiff {
    replacements: FxHashMap<SyntaxElement, SyntaxElement>,
    deletions: Vec<SyntaxElement>,
    // the vec as well as the indexmap are both here to preserve order
    insertions: FxIndexMap<SyntaxElement, Vec<SyntaxElement>>,
}

impl TreeDiff {
    pub fn into_text_edit(&self, builder: &mut TextEditBuilder) {
        for (anchor, to) in self.insertions.iter() {
            to.iter().for_each(|to| builder.insert(anchor.text_range().end(), to.to_string()));
        }
        for (from, to) in self.replacements.iter() {
            builder.replace(from.text_range(), to.to_string())
        }
        for text_range in self.deletions.iter().map(SyntaxElement::text_range) {
            builder.delete(text_range);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.replacements.is_empty() && self.deletions.is_empty() && self.insertions.is_empty()
    }
}

/// Finds minimal the diff, which, applied to `from`, will result in `to`.
///
/// Specifically, returns a structure that consists of a replacements, insertions and deletions
/// such that applying this map on `from` will result in `to`.
///
/// This function tries to find a fine-grained diff.
pub fn diff(from: &SyntaxNode, to: &SyntaxNode) -> TreeDiff {
    let mut diff = TreeDiff {
        replacements: FxHashMap::default(),
        insertions: FxIndexMap::default(),
        deletions: Vec::new(),
    };
    let (from, to) = (from.clone().into(), to.clone().into());

    // FIXME: this is horrible inefficient. I bet there's a cool algorithm to diff trees properly.
    if !syntax_element_eq(&from, &to) {
        go(&mut diff, from, to);
    }
    return diff;

    fn syntax_element_eq(lhs: &SyntaxElement, rhs: &SyntaxElement) -> bool {
        lhs.kind() == rhs.kind()
            && lhs.text_range().len() == rhs.text_range().len()
            && match (&lhs, &rhs) {
                (NodeOrToken::Node(lhs), NodeOrToken::Node(rhs)) => {
                    lhs.green() == rhs.green() || lhs.text() == rhs.text()
                }
                (NodeOrToken::Token(lhs), NodeOrToken::Token(rhs)) => lhs.text() == rhs.text(),
                _ => false,
            }
    }

    fn go(diff: &mut TreeDiff, lhs: SyntaxElement, rhs: SyntaxElement) {
        let (lhs, rhs) = match lhs.as_node().zip(rhs.as_node()) {
            Some((lhs, rhs)) => (lhs, rhs),
            _ => {
                mark::hit!(diff_node_token_replace);
                diff.replacements.insert(lhs, rhs);
                return;
            }
        };

        let mut rhs_children = rhs.children_with_tokens();
        let mut lhs_children = lhs.children_with_tokens();
        let mut last_lhs = None;
        loop {
            let lhs_child = lhs_children.next();
            match (lhs_child.clone(), rhs_children.next()) {
                (None, None) => break,
                (None, Some(element)) => match last_lhs.clone() {
                    Some(prev) => {
                        mark::hit!(diff_insert);
                        diff.insertions.entry(prev).or_insert_with(Vec::new).push(element);
                    }
                    // first iteration, this means we got no anchor element to insert after
                    // therefor replace the parent node instead
                    None => {
                        mark::hit!(diff_replace_parent);
                        diff.replacements.insert(lhs.clone().into(), rhs.clone().into());
                        break;
                    }
                },
                (Some(element), None) => {
                    mark::hit!(diff_delete);
                    diff.deletions.push(element);
                }
                (Some(ref lhs_ele), Some(ref rhs_ele)) if syntax_element_eq(lhs_ele, rhs_ele) => {}
                (Some(lhs_ele), Some(rhs_ele)) => go(diff, lhs_ele, rhs_ele),
            }
            last_lhs = lhs_child.or(last_lhs);
        }
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

#[derive(Debug, PartialEq, Eq, Hash)]
enum InsertPos {
    FirstChildOf(SyntaxNode),
    Before(SyntaxElement),
    After(SyntaxElement),
}

#[derive(Default)]
pub struct SyntaxRewriter<'a> {
    f: Option<Box<dyn Fn(&SyntaxElement) -> Option<SyntaxElement> + 'a>>,
    //FIXME: add debug_assertions that all elements are in fact from the same file.
    replacements: FxHashMap<SyntaxElement, Replacement>,
    insertions: IndexMap<InsertPos, Vec<SyntaxElement>>,
}

impl fmt::Debug for SyntaxRewriter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SyntaxRewriter").field("replacements", &self.replacements).finish()
    }
}

impl<'a> SyntaxRewriter<'a> {
    pub fn from_fn(f: impl Fn(&SyntaxElement) -> Option<SyntaxElement> + 'a) -> SyntaxRewriter<'a> {
        SyntaxRewriter {
            f: Some(Box::new(f)),
            replacements: FxHashMap::default(),
            insertions: IndexMap::default(),
        }
    }
    pub fn delete<T: Clone + Into<SyntaxElement>>(&mut self, what: &T) {
        let what = what.clone().into();
        let replacement = Replacement::Delete;
        self.replacements.insert(what, replacement);
    }
    pub fn insert_before<T: Clone + Into<SyntaxElement>, U: Clone + Into<SyntaxElement>>(
        &mut self,
        before: &T,
        what: &U,
    ) {
        self.insertions
            .entry(InsertPos::Before(before.clone().into()))
            .or_insert_with(Vec::new)
            .push(what.clone().into());
    }
    pub fn insert_after<T: Clone + Into<SyntaxElement>, U: Clone + Into<SyntaxElement>>(
        &mut self,
        after: &T,
        what: &U,
    ) {
        self.insertions
            .entry(InsertPos::After(after.clone().into()))
            .or_insert_with(Vec::new)
            .push(what.clone().into());
    }
    pub fn insert_as_first_child<T: Clone + Into<SyntaxNode>, U: Clone + Into<SyntaxElement>>(
        &mut self,
        parent: &T,
        what: &U,
    ) {
        self.insertions
            .entry(InsertPos::FirstChildOf(parent.clone().into()))
            .or_insert_with(Vec::new)
            .push(what.clone().into());
    }
    pub fn insert_many_before<
        T: Clone + Into<SyntaxElement>,
        U: IntoIterator<Item = SyntaxElement>,
    >(
        &mut self,
        before: &T,
        what: U,
    ) {
        self.insertions
            .entry(InsertPos::Before(before.clone().into()))
            .or_insert_with(Vec::new)
            .extend(what);
    }
    pub fn insert_many_after<
        T: Clone + Into<SyntaxElement>,
        U: IntoIterator<Item = SyntaxElement>,
    >(
        &mut self,
        after: &T,
        what: U,
    ) {
        self.insertions
            .entry(InsertPos::After(after.clone().into()))
            .or_insert_with(Vec::new)
            .extend(what);
    }
    pub fn insert_many_as_first_children<
        T: Clone + Into<SyntaxNode>,
        U: IntoIterator<Item = SyntaxElement>,
    >(
        &mut self,
        parent: &T,
        what: U,
    ) {
        self.insertions
            .entry(InsertPos::FirstChildOf(parent.clone().into()))
            .or_insert_with(Vec::new)
            .extend(what)
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
        if self.f.is_none() && self.replacements.is_empty() && self.insertions.is_empty() {
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
        fn element_to_node_or_parent(element: &SyntaxElement) -> SyntaxNode {
            match element {
                SyntaxElement::Node(it) => it.clone(),
                SyntaxElement::Token(it) => it.parent(),
            }
        }

        assert!(self.f.is_none());
        self.replacements
            .keys()
            .map(element_to_node_or_parent)
            .chain(self.insertions.keys().map(|pos| match pos {
                InsertPos::FirstChildOf(it) => it.clone(),
                InsertPos::Before(it) | InsertPos::After(it) => element_to_node_or_parent(it),
            }))
            // If we only have one replacement/insertion, we must return its parent node, since `rewrite` does
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

    fn insertions(&self, pos: &InsertPos) -> Option<impl Iterator<Item = SyntaxElement> + '_> {
        self.insertions.get(pos).map(|insertions| insertions.iter().cloned())
    }

    fn rewrite_children(&self, node: &SyntaxNode) -> SyntaxNode {
        //  FIXME: this could be made much faster.
        let mut new_children = Vec::new();
        if let Some(elements) = self.insertions(&InsertPos::FirstChildOf(node.clone())) {
            new_children.extend(elements.map(element_to_green));
        }
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
        if let Some(elements) = self.insertions(&InsertPos::Before(element.clone())) {
            acc.extend(elements.map(element_to_green));
        }
        if let Some(replacement) = self.replacement(&element) {
            match replacement {
                Replacement::Single(element) => acc.push(element_to_green(element)),
                Replacement::Many(replacements) => {
                    acc.extend(replacements.into_iter().map(element_to_green))
                }
                Replacement::Delete => (),
            };
        } else {
            match element {
                NodeOrToken::Token(it) => acc.push(NodeOrToken::Token(it.green().clone())),
                NodeOrToken::Node(it) => {
                    acc.push(NodeOrToken::Node(self.rewrite_children(it).green().clone()));
                }
            }
        }
        if let Some(elements) = self.insertions(&InsertPos::After(element.clone())) {
            acc.extend(elements.map(element_to_green));
        }
    }
}

fn element_to_green(element: SyntaxElement) -> NodeOrToken<rowan::GreenNode, rowan::GreenToken> {
    match element {
        NodeOrToken::Node(it) => NodeOrToken::Node(it.green().clone()),
        NodeOrToken::Token(it) => NodeOrToken::Token(it.green().clone()),
    }
}

impl ops::AddAssign for SyntaxRewriter<'_> {
    fn add_assign(&mut self, rhs: SyntaxRewriter) {
        assert!(rhs.f.is_none());
        self.replacements.extend(rhs.replacements);
        for (pos, insertions) in rhs.insertions.into_iter() {
            match self.insertions.entry(pos) {
                indexmap::map::Entry::Occupied(mut occupied) => {
                    occupied.get_mut().extend(insertions)
                }
                indexmap::map::Entry::Vacant(vacant) => drop(vacant.insert(insertions)),
            }
        }
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

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use itertools::Itertools;
    use parser::SyntaxKind;
    use test_utils::mark;
    use text_edit::TextEdit;

    use crate::{AstNode, SyntaxElement};

    #[test]
    fn replace_node_token() {
        mark::check!(diff_node_token_replace);
        check_diff(
            r#"use node;"#,
            r#"ident"#,
            expect![[r#"
                insertions:



                replacements:

                Line 0: Token(USE_KW@0..3 "use") -> ident

                deletions:

                Line 1: " "
                Line 1: node
                Line 1: ;
            "#]],
        );
    }

    #[test]
    fn insert() {
        mark::check!(diff_insert);
        check_diff(
            r#"use foo;"#,
            r#"use foo;
use bar;"#,
            expect![[r#"
                insertions:

                Line 0: Node(USE@0..8)
                -> "\n"
                -> use bar;

                replacements:



                deletions:


            "#]],
        );
    }

    #[test]
    fn replace_parent() {
        mark::check!(diff_replace_parent);
        check_diff(
            r#""#,
            r#"use foo::bar;"#,
            expect![[r#"
                insertions:



                replacements:

                Line 0: Node(SOURCE_FILE@0..0) -> use foo::bar;

                deletions:


            "#]],
        );
    }

    #[test]
    fn delete() {
        mark::check!(diff_delete);
        check_diff(
            r#"use foo;
            use bar;"#,
            r#"use foo;"#,
            expect![[r#"
                insertions:



                replacements:



                deletions:

                Line 1: "\n            "
                Line 2: use bar;
            "#]],
        );
    }

    #[test]
    fn insert_use() {
        check_diff(
            r#"
use expect_test::{expect, Expect};

use crate::AstNode;
"#,
            r#"
use expect_test::{expect, Expect};
use text_edit::TextEdit;

use crate::AstNode;
"#,
            expect![[r#"
                insertions:

                Line 4: Token(WHITESPACE@56..57 "\n")
                -> use crate::AstNode;
                -> "\n"

                replacements:

                Line 2: Token(WHITESPACE@35..37 "\n\n") -> "\n"
                Line 4: Token(CRATE_KW@41..46 "crate") -> text_edit
                Line 4: Token(IDENT@48..55 "AstNode") -> TextEdit
                Line 4: Token(WHITESPACE@56..57 "\n") -> "\n\n"

                deletions:


            "#]],
        )
    }

    #[test]
    fn remove_use() {
        check_diff(
            r#"
use expect_test::{expect, Expect};
use text_edit::TextEdit;

use crate::AstNode;
"#,
            r#"
use expect_test::{expect, Expect};

use crate::AstNode;
"#,
            expect![[r#"
                insertions:



                replacements:

                Line 2: Token(WHITESPACE@35..36 "\n") -> "\n\n"
                Line 3: Node(NAME_REF@40..49) -> crate
                Line 3: Token(IDENT@51..59 "TextEdit") -> AstNode
                Line 3: Token(WHITESPACE@60..62 "\n\n") -> "\n"

                deletions:

                Line 4: use crate::AstNode;
                Line 5: "\n"
            "#]],
        )
    }

    #[test]
    fn merge_use() {
        check_diff(
            r#"
use std::{
    fmt,
    hash::BuildHasherDefault,
    ops::{self, RangeInclusive},
};
"#,
            r#"
use std::fmt;
use std::hash::BuildHasherDefault;
use std::ops::{self, RangeInclusive};
"#,
            expect![[r#"
                insertions:

                Line 2: Node(PATH_SEGMENT@5..8)
                -> ::
                -> fmt
                Line 6: Token(WHITESPACE@86..87 "\n")
                -> use std::hash::BuildHasherDefault;
                -> "\n"
                -> use std::ops::{self, RangeInclusive};
                -> "\n"

                replacements:

                Line 2: Token(IDENT@5..8 "std") -> std

                deletions:

                Line 2: ::
                Line 2: {
                    fmt,
                    hash::BuildHasherDefault,
                    ops::{self, RangeInclusive},
                }
            "#]],
        )
    }

    #[test]
    fn early_return_assist() {
        check_diff(
            r#"
fn main() {
    if let Ok(x) = Err(92) {
        foo(x);
    }
}
            "#,
            r#"
fn main() {
    let x = match Err(92) {
        Ok(it) => it,
        _ => return,
    };
    foo(x);
}
            "#,
            expect![[r#"
                insertions:

                Line 3: Node(BLOCK_EXPR@40..63)
                -> " "
                -> match Err(92) {
                        Ok(it) => it,
                        _ => return,
                    }
                -> ;
                Line 5: Token(R_CURLY@64..65 "}")
                -> "\n"
                -> }

                replacements:

                Line 3: Token(IF_KW@17..19 "if") -> let
                Line 3: Token(LET_KW@20..23 "let") -> x
                Line 3: Node(BLOCK_EXPR@40..63) -> =
                Line 5: Token(WHITESPACE@63..64 "\n") -> "\n    "
                Line 5: Token(R_CURLY@64..65 "}") -> foo(x);

                deletions:

                Line 3: " "
                Line 3: Ok(x)
                Line 3: " "
                Line 3: =
                Line 3: " "
                Line 3: Err(92)
            "#]],
        )
    }

    fn check_diff(from: &str, to: &str, expected_diff: Expect) {
        let from_node = crate::SourceFile::parse(from).tree().syntax().clone();
        let to_node = crate::SourceFile::parse(to).tree().syntax().clone();
        let diff = super::diff(&from_node, &to_node);

        let line_number =
            |syn: &SyntaxElement| from[..syn.text_range().start().into()].lines().count();

        let fmt_syntax = |syn: &SyntaxElement| match syn.kind() {
            SyntaxKind::WHITESPACE => format!("{:?}", syn.to_string()),
            _ => format!("{}", syn),
        };

        let insertions = diff.insertions.iter().format_with("\n", |(k, v), f| {
            f(&format!(
                "Line {}: {:?}\n-> {}",
                line_number(k),
                k,
                v.iter().format_with("\n-> ", |v, f| f(&fmt_syntax(v)))
            ))
        });

        let replacements = diff
            .replacements
            .iter()
            .sorted_by_key(|(syntax, _)| syntax.text_range().start())
            .format_with("\n", |(k, v), f| {
                f(&format!("Line {}: {:?} -> {}", line_number(k), k, fmt_syntax(v)))
            });

        let deletions = diff
            .deletions
            .iter()
            .format_with("\n", |v, f| f(&format!("Line {}: {}", line_number(v), &fmt_syntax(v))));

        let actual = format!(
            "insertions:\n\n{}\n\nreplacements:\n\n{}\n\ndeletions:\n\n{}\n",
            insertions, replacements, deletions
        );
        expected_diff.assert_eq(&actual);

        let mut from = from.to_owned();
        let mut text_edit = TextEdit::builder();
        diff.into_text_edit(&mut text_edit);
        text_edit.finish().apply(&mut from);
        assert_eq!(&*from, to, "diff did not turn `from` to `to`");
    }
}
