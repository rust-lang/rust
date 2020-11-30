//! Handle syntactic aspects of inserting a new `use`.
use std::{cmp::Ordering, iter::successors};

use crate::RootDatabase;
use hir::Semantics;
use itertools::{EitherOrBoth, Itertools};
use syntax::{
    algo::SyntaxRewriter,
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make, AstNode, PathSegmentKind, VisibilityOwner,
    },
    AstToken, InsertPosition, NodeOrToken, SyntaxElement, SyntaxNode, SyntaxToken,
};
use test_utils::mark;

#[derive(Debug, Clone)]
pub enum ImportScope {
    File(ast::SourceFile),
    Module(ast::ItemList),
}

impl ImportScope {
    pub fn from(syntax: SyntaxNode) -> Option<Self> {
        if let Some(module) = ast::Module::cast(syntax.clone()) {
            module.item_list().map(ImportScope::Module)
        } else if let this @ Some(_) = ast::SourceFile::cast(syntax.clone()) {
            this.map(ImportScope::File)
        } else {
            ast::ItemList::cast(syntax).map(ImportScope::Module)
        }
    }

    /// Determines the containing syntax node in which to insert a `use` statement affecting `position`.
    pub fn find_insert_use_container(
        position: &SyntaxNode,
        sema: &Semantics<'_, RootDatabase>,
    ) -> Option<Self> {
        sema.ancestors_with_macros(position.clone()).find_map(Self::from)
    }

    pub fn as_syntax_node(&self) -> &SyntaxNode {
        match self {
            ImportScope::File(file) => file.syntax(),
            ImportScope::Module(item_list) => item_list.syntax(),
        }
    }

    fn indent_level(&self) -> IndentLevel {
        match self {
            ImportScope::File(file) => file.indent_level(),
            ImportScope::Module(item_list) => item_list.indent_level() + 1,
        }
    }

    fn first_insert_pos(&self) -> (InsertPosition<SyntaxElement>, AddBlankLine) {
        match self {
            ImportScope::File(_) => (InsertPosition::First, AddBlankLine::AfterTwice),
            // don't insert the imports before the item list's opening curly brace
            ImportScope::Module(item_list) => item_list
                .l_curly_token()
                .map(|b| (InsertPosition::After(b.into()), AddBlankLine::Around))
                .unwrap_or((InsertPosition::First, AddBlankLine::AfterTwice)),
        }
    }

    fn insert_pos_after_last_inner_element(&self) -> (InsertPosition<SyntaxElement>, AddBlankLine) {
        self.as_syntax_node()
            .children_with_tokens()
            .filter(|child| match child {
                NodeOrToken::Node(node) => is_inner_attribute(node.clone()),
                NodeOrToken::Token(token) => is_inner_comment(token.clone()),
            })
            .last()
            .map(|last_inner_element| {
                (InsertPosition::After(last_inner_element.into()), AddBlankLine::BeforeTwice)
            })
            .unwrap_or_else(|| self.first_insert_pos())
    }
}

fn is_inner_attribute(node: SyntaxNode) -> bool {
    ast::Attr::cast(node).map(|attr| attr.kind()) == Some(ast::AttrKind::Inner)
}

fn is_inner_comment(token: SyntaxToken) -> bool {
    ast::Comment::cast(token).and_then(|comment| comment.kind().doc)
        == Some(ast::CommentPlacement::Inner)
}

/// Insert an import path into the given file/node. A `merge` value of none indicates that no import merging is allowed to occur.
pub fn insert_use<'a>(
    scope: &ImportScope,
    path: ast::Path,
    merge: Option<MergeBehaviour>,
) -> SyntaxRewriter<'a> {
    let _p = profile::span("insert_use");
    let mut rewriter = SyntaxRewriter::default();
    let use_item = make::use_(make::use_tree(path.clone(), None, None, false));
    // merge into existing imports if possible
    if let Some(mb) = merge {
        for existing_use in scope.as_syntax_node().children().filter_map(ast::Use::cast) {
            if let Some(merged) = try_merge_imports(&existing_use, &use_item, mb) {
                rewriter.replace(existing_use.syntax(), merged.syntax());
                return rewriter;
            }
        }
    }

    // either we weren't allowed to merge or there is no import that fits the merge conditions
    // so look for the place we have to insert to
    let (insert_position, add_blank) = find_insert_position(scope, path);

    let indent = if let ident_level @ 1..=usize::MAX = scope.indent_level().0 as usize {
        Some(make::tokens::whitespace(&" ".repeat(4 * ident_level)).into())
    } else {
        None
    };

    let to_insert: Vec<SyntaxElement> = {
        let mut buf = Vec::new();

        match add_blank {
            AddBlankLine::Before | AddBlankLine::Around => {
                buf.push(make::tokens::single_newline().into())
            }
            AddBlankLine::BeforeTwice => buf.push(make::tokens::blank_line().into()),
            _ => (),
        }

        if add_blank.has_before() {
            if let Some(indent) = indent.clone() {
                mark::hit!(insert_use_indent_before);
                buf.push(indent);
            }
        }

        buf.push(use_item.syntax().clone().into());

        match add_blank {
            AddBlankLine::After | AddBlankLine::Around => {
                buf.push(make::tokens::single_newline().into())
            }
            AddBlankLine::AfterTwice => buf.push(make::tokens::blank_line().into()),
            _ => (),
        }

        // only add indentation *after* our stuff if there's another node directly after it
        if add_blank.has_after() && matches!(insert_position, InsertPosition::Before(_)) {
            if let Some(indent) = indent {
                mark::hit!(insert_use_indent_after);
                buf.push(indent);
            }
        } else if add_blank.has_after() && matches!(insert_position, InsertPosition::After(_)) {
            mark::hit!(insert_use_no_indent_after);
        }

        buf
    };

    match insert_position {
        InsertPosition::First => {
            rewriter.insert_many_as_first_children(scope.as_syntax_node(), to_insert)
        }
        InsertPosition::Last => return rewriter, // actually unreachable
        InsertPosition::Before(anchor) => rewriter.insert_many_before(&anchor, to_insert),
        InsertPosition::After(anchor) => rewriter.insert_many_after(&anchor, to_insert),
    }
    rewriter
}

fn eq_visibility(vis0: Option<ast::Visibility>, vis1: Option<ast::Visibility>) -> bool {
    match (vis0, vis1) {
        (None, None) => true,
        // FIXME: Don't use the string representation to check for equality
        // spaces inside of the node would break this comparison
        (Some(vis0), Some(vis1)) => vis0.to_string() == vis1.to_string(),
        _ => false,
    }
}

pub fn try_merge_imports(
    lhs: &ast::Use,
    rhs: &ast::Use,
    merge_behaviour: MergeBehaviour,
) -> Option<ast::Use> {
    // don't merge imports with different visibilities
    if !eq_visibility(lhs.visibility(), rhs.visibility()) {
        return None;
    }
    let lhs_tree = lhs.use_tree()?;
    let rhs_tree = rhs.use_tree()?;
    let merged = try_merge_trees(&lhs_tree, &rhs_tree, merge_behaviour)?;
    Some(lhs.with_use_tree(merged))
}

pub fn try_merge_trees(
    lhs: &ast::UseTree,
    rhs: &ast::UseTree,
    merge: MergeBehaviour,
) -> Option<ast::UseTree> {
    let lhs_path = lhs.path()?;
    let rhs_path = rhs.path()?;

    let (lhs_prefix, rhs_prefix) = common_prefix(&lhs_path, &rhs_path)?;
    let (lhs, rhs) = if is_simple_path(lhs)
        && is_simple_path(rhs)
        && lhs_path == lhs_prefix
        && rhs_path == rhs_prefix
    {
        (lhs.clone(), rhs.clone())
    } else {
        (lhs.split_prefix(&lhs_prefix), rhs.split_prefix(&rhs_prefix))
    };
    recursive_merge(&lhs, &rhs, merge)
}

/// Recursively "zips" together lhs and rhs.
fn recursive_merge(
    lhs: &ast::UseTree,
    rhs: &ast::UseTree,
    merge: MergeBehaviour,
) -> Option<ast::UseTree> {
    let mut use_trees = lhs
        .use_tree_list()
        .into_iter()
        .flat_map(|list| list.use_trees())
        // we use Option here to early return from this function(this is not the same as a `filter` op)
        .map(|tree| match merge.is_tree_allowed(&tree) {
            true => Some(tree),
            false => None,
        })
        .collect::<Option<Vec<_>>>()?;
    use_trees.sort_unstable_by(|a, b| path_cmp_for_sort(a.path(), b.path()));
    for rhs_t in rhs.use_tree_list().into_iter().flat_map(|list| list.use_trees()) {
        if !merge.is_tree_allowed(&rhs_t) {
            return None;
        }
        let rhs_path = rhs_t.path();
        match use_trees.binary_search_by(|lhs_t| {
            let (lhs_t, rhs_t) = match lhs_t
                .path()
                .zip(rhs_path.clone())
                .and_then(|(lhs, rhs)| common_prefix(&lhs, &rhs))
            {
                Some((lhs_p, rhs_p)) => (lhs_t.split_prefix(&lhs_p), rhs_t.split_prefix(&rhs_p)),
                None => (lhs_t.clone(), rhs_t.clone()),
            };

            path_cmp_bin_search(lhs_t.path(), rhs_t.path())
        }) {
            Ok(idx) => {
                let lhs_t = &mut use_trees[idx];
                let lhs_path = lhs_t.path()?;
                let rhs_path = rhs_path?;
                let (lhs_prefix, rhs_prefix) = common_prefix(&lhs_path, &rhs_path)?;
                if lhs_prefix == lhs_path && rhs_prefix == rhs_path {
                    let tree_is_self = |tree: ast::UseTree| {
                        tree.path().as_ref().map(path_is_self).unwrap_or(false)
                    };
                    // check if only one of the two trees has a tree list, and whether that then contains `self` or not.
                    // If this is the case we can skip this iteration since the path without the list is already included in the other one via `self`
                    let tree_contains_self = |tree: &ast::UseTree| {
                        tree.use_tree_list()
                            .map(|tree_list| tree_list.use_trees().any(tree_is_self))
                            .unwrap_or(false)
                    };
                    match (tree_contains_self(&lhs_t), tree_contains_self(&rhs_t)) {
                        (true, false) => continue,
                        (false, true) => {
                            *lhs_t = rhs_t;
                            continue;
                        }
                        _ => (),
                    }

                    // glob imports arent part of the use-tree lists so we need to special handle them here as well
                    // this special handling is only required for when we merge a module import into a glob import of said module
                    // see the `merge_self_glob` or `merge_mod_into_glob` tests
                    if lhs_t.star_token().is_some() || rhs_t.star_token().is_some() {
                        *lhs_t = make::use_tree(
                            make::path_unqualified(make::path_segment_self()),
                            None,
                            None,
                            false,
                        );
                        use_trees.insert(idx, make::glob_use_tree());
                        continue;
                    }

                    if lhs_t.use_tree_list().is_none() && rhs_t.use_tree_list().is_none() {
                        continue;
                    }
                }
                let lhs = lhs_t.split_prefix(&lhs_prefix);
                let rhs = rhs_t.split_prefix(&rhs_prefix);
                match recursive_merge(&lhs, &rhs, merge) {
                    Some(use_tree) => use_trees[idx] = use_tree,
                    None => return None,
                }
            }
            Err(_)
                if merge == MergeBehaviour::Last
                    && use_trees.len() > 0
                    && rhs_t.use_tree_list().is_some() =>
            {
                return None
            }
            Err(idx) => {
                use_trees.insert(idx, rhs_t);
            }
        }
    }
    Some(lhs.with_use_tree_list(make::use_tree_list(use_trees)))
}

/// Traverses both paths until they differ, returning the common prefix of both.
fn common_prefix(lhs: &ast::Path, rhs: &ast::Path) -> Option<(ast::Path, ast::Path)> {
    let mut res = None;
    let mut lhs_curr = first_path(&lhs);
    let mut rhs_curr = first_path(&rhs);
    loop {
        match (lhs_curr.segment(), rhs_curr.segment()) {
            (Some(lhs), Some(rhs)) if lhs.syntax().text() == rhs.syntax().text() => (),
            _ => break res,
        }
        res = Some((lhs_curr.clone(), rhs_curr.clone()));

        match lhs_curr.parent_path().zip(rhs_curr.parent_path()) {
            Some((lhs, rhs)) => {
                lhs_curr = lhs;
                rhs_curr = rhs;
            }
            _ => break res,
        }
    }
}

fn is_simple_path(use_tree: &ast::UseTree) -> bool {
    use_tree.use_tree_list().is_none() && use_tree.star_token().is_none()
}

fn path_is_self(path: &ast::Path) -> bool {
    path.segment().and_then(|seg| seg.self_token()).is_some() && path.qualifier().is_none()
}

#[inline]
fn first_segment(path: &ast::Path) -> Option<ast::PathSegment> {
    first_path(path).segment()
}

fn first_path(path: &ast::Path) -> ast::Path {
    successors(Some(path.clone()), ast::Path::qualifier).last().unwrap()
}

fn segment_iter(path: &ast::Path) -> impl Iterator<Item = ast::PathSegment> + Clone {
    // cant make use of SyntaxNode::siblings, because the returned Iterator is not clone
    successors(first_segment(path), |p| p.parent_path().parent_path().and_then(|p| p.segment()))
}

fn path_len(path: ast::Path) -> usize {
    segment_iter(&path).count()
}

/// Orders paths in the following way:
/// the sole self token comes first, after that come uppercase identifiers, then lowercase identifiers
// FIXME: rustfmt sorts lowercase idents before uppercase, in general we want to have the same ordering rustfmt has
// which is `self` and `super` first, then identifier imports with lowercase ones first, then glob imports and at last list imports.
// Example foo::{self, foo, baz, Baz, Qux, *, {Bar}}
fn path_cmp_for_sort(a: Option<ast::Path>, b: Option<ast::Path>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(ref a), Some(ref b)) => match (path_is_self(a), path_is_self(b)) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => path_cmp_short(a, b),
        },
    }
}

/// Path comparison func for binary searching for merging.
fn path_cmp_bin_search(lhs: Option<ast::Path>, rhs: Option<ast::Path>) -> Ordering {
    match (lhs.as_ref().and_then(first_segment), rhs.as_ref().and_then(first_segment)) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(ref a), Some(ref b)) => path_segment_cmp(a, b),
    }
}

/// Short circuiting comparison, if both paths are equal until one of them ends they are considered
/// equal
fn path_cmp_short(a: &ast::Path, b: &ast::Path) -> Ordering {
    let a = segment_iter(a);
    let b = segment_iter(b);
    // cmp_by would be useful for us here but that is currently unstable
    // cmp doesnt work due the lifetimes on text's return type
    a.zip(b)
        .find_map(|(a, b)| match path_segment_cmp(&a, &b) {
            Ordering::Equal => None,
            ord => Some(ord),
        })
        .unwrap_or(Ordering::Equal)
}

/// Compares to paths, if one ends earlier than the other the has_tl parameters decide which is
/// greater as a a path that has a tree list should be greater, while one that just ends without
/// a tree list should be considered less.
fn use_tree_path_cmp(a: &ast::Path, a_has_tl: bool, b: &ast::Path, b_has_tl: bool) -> Ordering {
    let a_segments = segment_iter(a);
    let b_segments = segment_iter(b);
    // cmp_by would be useful for us here but that is currently unstable
    // cmp doesnt work due the lifetimes on text's return type
    a_segments
        .zip_longest(b_segments)
        .find_map(|zipped| match zipped {
            EitherOrBoth::Both(ref a, ref b) => match path_segment_cmp(a, b) {
                Ordering::Equal => None,
                ord => Some(ord),
            },
            EitherOrBoth::Left(_) if !b_has_tl => Some(Ordering::Greater),
            EitherOrBoth::Left(_) => Some(Ordering::Less),
            EitherOrBoth::Right(_) if !a_has_tl => Some(Ordering::Less),
            EitherOrBoth::Right(_) => Some(Ordering::Greater),
        })
        .unwrap_or(Ordering::Equal)
}

fn path_segment_cmp(a: &ast::PathSegment, b: &ast::PathSegment) -> Ordering {
    let a = a.name_ref();
    let b = b.name_ref();
    a.as_ref().map(ast::NameRef::text).cmp(&b.as_ref().map(ast::NameRef::text))
}

/// What type of merges are allowed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MergeBehaviour {
    /// Merge everything together creating deeply nested imports.
    Full,
    /// Only merge the last import level, doesn't allow import nesting.
    Last,
}

impl MergeBehaviour {
    #[inline]
    fn is_tree_allowed(&self, tree: &ast::UseTree) -> bool {
        match self {
            MergeBehaviour::Full => true,
            // only simple single segment paths are allowed
            MergeBehaviour::Last => {
                tree.use_tree_list().is_none() && tree.path().map(path_len) <= Some(1)
            }
        }
    }
}

#[derive(Eq, PartialEq, PartialOrd, Ord)]
enum ImportGroup {
    // the order here defines the order of new group inserts
    Std,
    ExternCrate,
    ThisCrate,
    ThisModule,
    SuperModule,
}

impl ImportGroup {
    fn new(path: &ast::Path) -> ImportGroup {
        let default = ImportGroup::ExternCrate;

        let first_segment = match first_segment(path) {
            Some(it) => it,
            None => return default,
        };

        let kind = first_segment.kind().unwrap_or(PathSegmentKind::SelfKw);
        match kind {
            PathSegmentKind::SelfKw => ImportGroup::ThisModule,
            PathSegmentKind::SuperKw => ImportGroup::SuperModule,
            PathSegmentKind::CrateKw => ImportGroup::ThisCrate,
            PathSegmentKind::Name(name) => match name.text().as_str() {
                "std" => ImportGroup::Std,
                "core" => ImportGroup::Std,
                _ => ImportGroup::ExternCrate,
            },
            PathSegmentKind::Type { .. } => unreachable!(),
        }
    }
}

#[derive(PartialEq, Eq)]
enum AddBlankLine {
    Before,
    BeforeTwice,
    Around,
    After,
    AfterTwice,
}

impl AddBlankLine {
    fn has_before(&self) -> bool {
        matches!(self, AddBlankLine::Before | AddBlankLine::BeforeTwice | AddBlankLine::Around)
    }
    fn has_after(&self) -> bool {
        matches!(self, AddBlankLine::After | AddBlankLine::AfterTwice | AddBlankLine::Around)
    }
}

fn find_insert_position(
    scope: &ImportScope,
    insert_path: ast::Path,
) -> (InsertPosition<SyntaxElement>, AddBlankLine) {
    let group = ImportGroup::new(&insert_path);
    let path_node_iter = scope
        .as_syntax_node()
        .children()
        .filter_map(|node| ast::Use::cast(node.clone()).zip(Some(node)))
        .flat_map(|(use_, node)| {
            let tree = use_.use_tree()?;
            let path = tree.path()?;
            let has_tl = tree.use_tree_list().is_some();
            Some((path, has_tl, node))
        });
    // Iterator that discards anything thats not in the required grouping
    // This implementation allows the user to rearrange their import groups as this only takes the first group that fits
    let group_iter = path_node_iter
        .clone()
        .skip_while(|(path, ..)| ImportGroup::new(path) != group)
        .take_while(|(path, ..)| ImportGroup::new(path) == group);

    // track the last element we iterated over, if this is still None after the iteration then that means we never iterated in the first place
    let mut last = None;
    // find the element that would come directly after our new import
    let post_insert = group_iter.inspect(|(.., node)| last = Some(node.clone())).find(
        |&(ref path, has_tl, _)| {
            use_tree_path_cmp(&insert_path, false, path, has_tl) != Ordering::Greater
        },
    );
    match post_insert {
        // insert our import before that element
        Some((.., node)) => (InsertPosition::Before(node.into()), AddBlankLine::After),
        // there is no element after our new import, so append it to the end of the group
        None => match last {
            Some(node) => (InsertPosition::After(node.into()), AddBlankLine::Before),
            // the group we were looking for actually doesnt exist, so insert
            None => {
                // similar concept here to the `last` from above
                let mut last = None;
                // find the group that comes after where we want to insert
                let post_group = path_node_iter
                    .inspect(|(.., node)| last = Some(node.clone()))
                    .find(|(p, ..)| ImportGroup::new(p) > group);
                match post_group {
                    Some((.., node)) => {
                        (InsertPosition::Before(node.into()), AddBlankLine::AfterTwice)
                    }
                    // there is no such group, so append after the last one
                    None => match last {
                        Some(node) => {
                            (InsertPosition::After(node.into()), AddBlankLine::BeforeTwice)
                        }
                        // there are no imports in this file at all
                        None => scope.insert_pos_after_last_inner_element(),
                    },
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use test_utils::assert_eq_text;

    #[test]
    fn insert_existing() {
        check_full("std::fs", "use std::fs;", "use std::fs;")
    }

    #[test]
    fn insert_start() {
        check_none(
            "std::bar::AA",
            r"
use std::bar::B;
use std::bar::D;
use std::bar::F;
use std::bar::G;",
            r"
use std::bar::AA;
use std::bar::B;
use std::bar::D;
use std::bar::F;
use std::bar::G;",
        )
    }

    #[test]
    fn insert_start_indent() {
        mark::check!(insert_use_indent_after);
        check_none(
            "std::bar::AA",
            r"
    use std::bar::B;
    use std::bar::D;",
            r"
    use std::bar::AA;
    use std::bar::B;
    use std::bar::D;",
        )
    }

    #[test]
    fn insert_middle() {
        check_none(
            "std::bar::EE",
            r"
use std::bar::A;
use std::bar::D;
use std::bar::F;
use std::bar::G;",
            r"
use std::bar::A;
use std::bar::D;
use std::bar::EE;
use std::bar::F;
use std::bar::G;",
        )
    }

    #[test]
    fn insert_middle_indent() {
        check_none(
            "std::bar::EE",
            r"
    use std::bar::A;
    use std::bar::D;
    use std::bar::F;
    use std::bar::G;",
            r"
    use std::bar::A;
    use std::bar::D;
    use std::bar::EE;
    use std::bar::F;
    use std::bar::G;",
        )
    }

    #[test]
    fn insert_end() {
        check_none(
            "std::bar::ZZ",
            r"
use std::bar::A;
use std::bar::D;
use std::bar::F;
use std::bar::G;",
            r"
use std::bar::A;
use std::bar::D;
use std::bar::F;
use std::bar::G;
use std::bar::ZZ;",
        )
    }

    #[test]
    fn insert_end_indent() {
        mark::check!(insert_use_indent_before);
        check_none(
            "std::bar::ZZ",
            r"
    use std::bar::A;
    use std::bar::D;
    use std::bar::F;
    use std::bar::G;",
            r"
    use std::bar::A;
    use std::bar::D;
    use std::bar::F;
    use std::bar::G;
    use std::bar::ZZ;",
        )
    }

    #[test]
    fn insert_middle_nested() {
        check_none(
            "std::bar::EE",
            r"
use std::bar::A;
use std::bar::{D, Z}; // example of weird imports due to user
use std::bar::F;
use std::bar::G;",
            r"
use std::bar::A;
use std::bar::EE;
use std::bar::{D, Z}; // example of weird imports due to user
use std::bar::F;
use std::bar::G;",
        )
    }

    #[test]
    fn insert_middle_groups() {
        check_none(
            "foo::bar::GG",
            r"
    use std::bar::A;
    use std::bar::D;

    use foo::bar::F;
    use foo::bar::H;",
            r"
    use std::bar::A;
    use std::bar::D;

    use foo::bar::F;
    use foo::bar::GG;
    use foo::bar::H;",
        )
    }

    #[test]
    fn insert_first_matching_group() {
        check_none(
            "foo::bar::GG",
            r"
    use foo::bar::A;
    use foo::bar::D;

    use std;

    use foo::bar::F;
    use foo::bar::H;",
            r"
    use foo::bar::A;
    use foo::bar::D;
    use foo::bar::GG;

    use std;

    use foo::bar::F;
    use foo::bar::H;",
        )
    }

    #[test]
    fn insert_missing_group_std() {
        check_none(
            "std::fmt",
            r"
    use foo::bar::A;
    use foo::bar::D;",
            r"
    use std::fmt;

    use foo::bar::A;
    use foo::bar::D;",
        )
    }

    #[test]
    fn insert_missing_group_self() {
        check_none(
            "self::fmt",
            r"
use foo::bar::A;
use foo::bar::D;",
            r"
use foo::bar::A;
use foo::bar::D;

use self::fmt;",
        )
    }

    #[test]
    fn insert_no_imports() {
        check_full(
            "foo::bar",
            "fn main() {}",
            r"use foo::bar;

fn main() {}",
        )
    }

    #[test]
    fn insert_empty_file() {
        // empty files will get two trailing newlines
        // this is due to the test case insert_no_imports above
        check_full(
            "foo::bar",
            "",
            r"use foo::bar;

",
        )
    }

    #[test]
    fn insert_empty_module() {
        mark::check!(insert_use_no_indent_after);
        check(
            "foo::bar",
            "mod x {}",
            r"{
    use foo::bar;
}",
            None,
            true,
        )
    }

    #[test]
    fn insert_after_inner_attr() {
        check_full(
            "foo::bar",
            r"#![allow(unused_imports)]",
            r"#![allow(unused_imports)]

use foo::bar;",
        )
    }

    #[test]
    fn insert_after_inner_attr2() {
        check_full(
            "foo::bar",
            r"#![allow(unused_imports)]

#![no_std]
fn main() {}",
            r"#![allow(unused_imports)]

#![no_std]

use foo::bar;
fn main() {}",
        );
    }

    #[test]
    fn inserts_after_single_line_inner_comments() {
        check_none(
            "foo::bar::Baz",
            "//! Single line inner comments do not allow any code before them.",
            r#"//! Single line inner comments do not allow any code before them.

use foo::bar::Baz;"#,
        );
    }

    #[test]
    fn inserts_after_multiline_inner_comments() {
        check_none(
            "foo::bar::Baz",
            r#"/*! Multiline inner comments do not allow any code before them. */

/*! Still an inner comment, cannot place any code before. */
fn main() {}"#,
            r#"/*! Multiline inner comments do not allow any code before them. */

/*! Still an inner comment, cannot place any code before. */

use foo::bar::Baz;
fn main() {}"#,
        )
    }

    #[test]
    fn inserts_after_all_inner_items() {
        check_none(
            "foo::bar::Baz",
            r#"#![allow(unused_imports)]
/*! Multiline line comment 2 */


//! Single line comment 1
#![no_std]
//! Single line comment 2
fn main() {}"#,
            r#"#![allow(unused_imports)]
/*! Multiline line comment 2 */


//! Single line comment 1
#![no_std]
//! Single line comment 2

use foo::bar::Baz;
fn main() {}"#,
        )
    }

    #[test]
    fn merge_groups() {
        check_last("std::io", r"use std::fmt;", r"use std::{fmt, io};")
    }

    #[test]
    fn merge_groups_last() {
        check_last(
            "std::io",
            r"use std::fmt::{Result, Display};",
            r"use std::fmt::{Result, Display};
use std::io;",
        )
    }

    #[test]
    fn merge_last_into_self() {
        check_last("foo::bar::baz", r"use foo::bar;", r"use foo::bar::{self, baz};");
    }

    #[test]
    fn merge_groups_full() {
        check_full(
            "std::io",
            r"use std::fmt::{Result, Display};",
            r"use std::{fmt::{Result, Display}, io};",
        )
    }

    #[test]
    fn merge_groups_long_full() {
        check_full(
            "std::foo::bar::Baz",
            r"use std::foo::bar::Qux;",
            r"use std::foo::bar::{Baz, Qux};",
        )
    }

    #[test]
    fn merge_groups_long_last() {
        check_last(
            "std::foo::bar::Baz",
            r"use std::foo::bar::Qux;",
            r"use std::foo::bar::{Baz, Qux};",
        )
    }

    #[test]
    fn merge_groups_long_full_list() {
        check_full(
            "std::foo::bar::Baz",
            r"use std::foo::bar::{Qux, Quux};",
            r"use std::foo::bar::{Baz, Quux, Qux};",
        )
    }

    #[test]
    fn merge_groups_long_last_list() {
        check_last(
            "std::foo::bar::Baz",
            r"use std::foo::bar::{Qux, Quux};",
            r"use std::foo::bar::{Baz, Quux, Qux};",
        )
    }

    #[test]
    fn merge_groups_long_full_nested() {
        check_full(
            "std::foo::bar::Baz",
            r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
            r"use std::foo::bar::{Baz, Qux, quux::{Fez, Fizz}};",
        )
    }

    #[test]
    fn merge_groups_long_last_nested() {
        check_last(
            "std::foo::bar::Baz",
            r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
            r"use std::foo::bar::Baz;
use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
        )
    }

    #[test]
    fn merge_groups_full_nested_deep() {
        check_full(
            "std::foo::bar::quux::Baz",
            r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
            r"use std::foo::bar::{Qux, quux::{Baz, Fez, Fizz}};",
        )
    }

    #[test]
    fn merge_groups_full_nested_long() {
        check_full(
            "std::foo::bar::Baz",
            r"use std::{foo::bar::Qux};",
            r"use std::{foo::bar::{Baz, Qux}};",
        );
    }

    #[test]
    fn merge_groups_last_nested_long() {
        check_full(
            "std::foo::bar::Baz",
            r"use std::{foo::bar::Qux};",
            r"use std::{foo::bar::{Baz, Qux}};",
        );
    }

    #[test]
    fn merge_groups_skip_pub() {
        check_full(
            "std::io",
            r"pub use std::fmt::{Result, Display};",
            r"pub use std::fmt::{Result, Display};
use std::io;",
        )
    }

    #[test]
    fn merge_groups_skip_pub_crate() {
        check_full(
            "std::io",
            r"pub(crate) use std::fmt::{Result, Display};",
            r"pub(crate) use std::fmt::{Result, Display};
use std::io;",
        )
    }

    #[test]
    #[ignore] // FIXME: Support this
    fn split_out_merge() {
        check_last(
            "std::fmt::Result",
            r"use std::{fmt, io};",
            r"use std::fmt::{self, Result};
use std::io;",
        )
    }

    #[test]
    fn merge_into_module_import() {
        check_full(
            "std::fmt::Result",
            r"use std::{fmt, io};",
            r"use std::{fmt::{self, Result}, io};",
        )
    }

    #[test]
    fn merge_groups_self() {
        check_full("std::fmt::Debug", r"use std::fmt;", r"use std::fmt::{self, Debug};")
    }

    #[test]
    fn merge_mod_into_glob() {
        check_full(
            "token::TokenKind",
            r"use token::TokenKind::*;",
            r"use token::TokenKind::{*, self};",
        )
        // FIXME: have it emit `use token::TokenKind::{self, *}`?
    }

    #[test]
    fn merge_self_glob() {
        check_full("self", r"use self::*;", r"use self::{*, self};")
        // FIXME: have it emit `use {self, *}`?
    }

    #[test]
    fn merge_glob_nested() {
        check_full(
            "foo::bar::quux::Fez",
            r"use foo::bar::{Baz, quux::*};",
            r"use foo::bar::{Baz, quux::{self::*, Fez}};",
        )
    }

    #[test]
    fn merge_nested_considers_first_segments() {
        check_full(
            "hir_ty::display::write_bounds_like_dyn_trait",
            r"use hir_ty::{autoderef, display::{HirDisplayError, HirFormatter}, method_resolution};",
            r"use hir_ty::{autoderef, display::{HirDisplayError, HirFormatter, write_bounds_like_dyn_trait}, method_resolution};",
        );
    }

    #[test]
    fn skip_merge_last_too_long() {
        check_last(
            "foo::bar",
            r"use foo::bar::baz::Qux;",
            r"use foo::bar;
use foo::bar::baz::Qux;",
        );
    }

    #[test]
    fn skip_merge_last_too_long2() {
        check_last(
            "foo::bar::baz::Qux",
            r"use foo::bar;",
            r"use foo::bar;
use foo::bar::baz::Qux;",
        );
    }

    #[test]
    fn insert_short_before_long() {
        check_none(
            "foo::bar",
            r"use foo::bar::baz::Qux;",
            r"use foo::bar;
use foo::bar::baz::Qux;",
        );
    }

    #[test]
    fn merge_last_fail() {
        check_merge_only_fail(
            r"use foo::bar::{baz::{Qux, Fez}};",
            r"use foo::bar::{baaz::{Quux, Feez}};",
            MergeBehaviour::Last,
        );
    }

    #[test]
    fn merge_last_fail1() {
        check_merge_only_fail(
            r"use foo::bar::{baz::{Qux, Fez}};",
            r"use foo::bar::baaz::{Quux, Feez};",
            MergeBehaviour::Last,
        );
    }

    #[test]
    fn merge_last_fail2() {
        check_merge_only_fail(
            r"use foo::bar::baz::{Qux, Fez};",
            r"use foo::bar::{baaz::{Quux, Feez}};",
            MergeBehaviour::Last,
        );
    }

    #[test]
    fn merge_last_fail3() {
        check_merge_only_fail(
            r"use foo::bar::baz::{Qux, Fez};",
            r"use foo::bar::baaz::{Quux, Feez};",
            MergeBehaviour::Last,
        );
    }

    fn check(
        path: &str,
        ra_fixture_before: &str,
        ra_fixture_after: &str,
        mb: Option<MergeBehaviour>,
        module: bool,
    ) {
        let mut syntax = ast::SourceFile::parse(ra_fixture_before).tree().syntax().clone();
        if module {
            syntax = syntax.descendants().find_map(ast::Module::cast).unwrap().syntax().clone();
        }
        let file = super::ImportScope::from(syntax).unwrap();
        let path = ast::SourceFile::parse(&format!("use {};", path))
            .tree()
            .syntax()
            .descendants()
            .find_map(ast::Path::cast)
            .unwrap();

        let rewriter = insert_use(&file, path, mb);
        let result = rewriter.rewrite(file.as_syntax_node()).to_string();
        assert_eq_text!(&result, ra_fixture_after);
    }

    fn check_full(path: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
        check(path, ra_fixture_before, ra_fixture_after, Some(MergeBehaviour::Full), false)
    }

    fn check_last(path: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
        check(path, ra_fixture_before, ra_fixture_after, Some(MergeBehaviour::Last), false)
    }

    fn check_none(path: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
        check(path, ra_fixture_before, ra_fixture_after, None, false)
    }

    fn check_merge_only_fail(ra_fixture0: &str, ra_fixture1: &str, mb: MergeBehaviour) {
        let use0 = ast::SourceFile::parse(ra_fixture0)
            .tree()
            .syntax()
            .descendants()
            .find_map(ast::Use::cast)
            .unwrap();

        let use1 = ast::SourceFile::parse(ra_fixture1)
            .tree()
            .syntax()
            .descendants()
            .find_map(ast::Use::cast)
            .unwrap();

        let result = try_merge_imports(&use0, &use1, mb);
        assert_eq!(result.map(|u| u.to_string()), None);
    }
}
