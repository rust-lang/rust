//! Handle syntactic aspects of merging UseTrees.
use std::cmp::Ordering;

use itertools::{EitherOrBoth, Itertools};
use syntax::{
    ast::{self, AstNode, HasAttrs, HasName, HasVisibility, PathSegmentKind},
    ted, TokenText,
};

use crate::syntax_helpers::node_ext::vis_eq;

/// What type of merges are allowed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MergeBehavior {
    /// Merge imports from the same crate into a single use statement.
    Crate,
    /// Merge imports from the same module into a single use statement.
    Module,
}

impl MergeBehavior {
    fn is_tree_allowed(&self, tree: &ast::UseTree) -> bool {
        match self {
            MergeBehavior::Crate => true,
            // only simple single segment paths are allowed
            MergeBehavior::Module => {
                tree.use_tree_list().is_none() && tree.path().map(path_len) <= Some(1)
            }
        }
    }
}

/// Merge `rhs` into `lhs` keeping both intact.
/// Returned AST is mutable.
pub fn try_merge_imports(
    lhs: &ast::Use,
    rhs: &ast::Use,
    merge_behavior: MergeBehavior,
) -> Option<ast::Use> {
    // don't merge imports with different visibilities
    if !eq_visibility(lhs.visibility(), rhs.visibility()) {
        return None;
    }
    if !eq_attrs(lhs.attrs(), rhs.attrs()) {
        return None;
    }

    let lhs = lhs.clone_subtree().clone_for_update();
    let rhs = rhs.clone_subtree().clone_for_update();
    let lhs_tree = lhs.use_tree()?;
    let rhs_tree = rhs.use_tree()?;
    try_merge_trees_mut(&lhs_tree, &rhs_tree, merge_behavior)?;
    Some(lhs)
}

/// Merge `rhs` into `lhs` keeping both intact.
/// Returned AST is mutable.
pub fn try_merge_trees(
    lhs: &ast::UseTree,
    rhs: &ast::UseTree,
    merge: MergeBehavior,
) -> Option<ast::UseTree> {
    let lhs = lhs.clone_subtree().clone_for_update();
    let rhs = rhs.clone_subtree().clone_for_update();
    try_merge_trees_mut(&lhs, &rhs, merge)?;
    Some(lhs)
}

fn try_merge_trees_mut(lhs: &ast::UseTree, rhs: &ast::UseTree, merge: MergeBehavior) -> Option<()> {
    let lhs_path = lhs.path()?;
    let rhs_path = rhs.path()?;

    let (lhs_prefix, rhs_prefix) = common_prefix(&lhs_path, &rhs_path)?;
    if !(lhs.is_simple_path()
        && rhs.is_simple_path()
        && lhs_path == lhs_prefix
        && rhs_path == rhs_prefix)
    {
        lhs.split_prefix(&lhs_prefix);
        rhs.split_prefix(&rhs_prefix);
    } else {
        ted::replace(lhs.syntax(), rhs.syntax());
        // we can safely return here, in this case `recursive_merge` doesn't do anything
        return Some(());
    }
    recursive_merge(lhs, rhs, merge)
}

/// Recursively merges rhs to lhs
#[must_use]
fn recursive_merge(lhs: &ast::UseTree, rhs: &ast::UseTree, merge: MergeBehavior) -> Option<()> {
    let mut use_trees: Vec<ast::UseTree> = lhs
        .use_tree_list()
        .into_iter()
        .flat_map(|list| list.use_trees())
        // We use Option here to early return from this function(this is not the
        // same as a `filter` op).
        .map(|tree| merge.is_tree_allowed(&tree).then_some(tree))
        .collect::<Option<_>>()?;
    use_trees.sort_unstable_by(|a, b| use_tree_cmp(a, b));
    for rhs_t in rhs.use_tree_list().into_iter().flat_map(|list| list.use_trees()) {
        if !merge.is_tree_allowed(&rhs_t) {
            return None;
        }
        let rhs_path = rhs_t.path();

        match use_trees
            .binary_search_by(|lhs_t| path_cmp_bin_search(lhs_t.path(), rhs_path.as_ref()))
        {
            Ok(idx) => {
                let lhs_t = &mut use_trees[idx];
                let lhs_path = lhs_t.path()?;
                let rhs_path = rhs_path?;
                let (lhs_prefix, rhs_prefix) = common_prefix(&lhs_path, &rhs_path)?;
                if lhs_prefix == lhs_path && rhs_prefix == rhs_path {
                    let tree_is_self = |tree: &ast::UseTree| {
                        tree.path().as_ref().map(path_is_self).unwrap_or(false)
                    };
                    // Check if only one of the two trees has a tree list, and
                    // whether that then contains `self` or not. If this is the
                    // case we can skip this iteration since the path without
                    // the list is already included in the other one via `self`.
                    let tree_contains_self = |tree: &ast::UseTree| {
                        tree.use_tree_list()
                            .map(|tree_list| tree_list.use_trees().any(|it| tree_is_self(&it)))
                            // Glob imports aren't part of the use-tree lists,
                            // so they need to be handled explicitly
                            .or_else(|| tree.star_token().map(|_| false))
                    };

                    if lhs_t.rename().and_then(|x| x.underscore_token()).is_some() {
                        ted::replace(lhs_t.syntax(), rhs_t.syntax());
                        *lhs_t = rhs_t;
                        continue;
                    }

                    match (tree_contains_self(lhs_t), tree_contains_self(&rhs_t)) {
                        (Some(true), None) => continue,
                        (None, Some(true)) => {
                            ted::replace(lhs_t.syntax(), rhs_t.syntax());
                            *lhs_t = rhs_t;
                            continue;
                        }
                        _ => (),
                    }

                    if lhs_t.is_simple_path() && rhs_t.is_simple_path() {
                        continue;
                    }
                }
                lhs_t.split_prefix(&lhs_prefix);
                rhs_t.split_prefix(&rhs_prefix);
                recursive_merge(lhs_t, &rhs_t, merge)?;
            }
            Err(_)
                if merge == MergeBehavior::Module
                    && !use_trees.is_empty()
                    && rhs_t.use_tree_list().is_some() =>
            {
                return None
            }
            Err(idx) => {
                use_trees.insert(idx, rhs_t.clone());
                lhs.get_or_create_use_tree_list().add_use_tree(rhs_t);
            }
        }
    }
    Some(())
}

/// Traverses both paths until they differ, returning the common prefix of both.
pub fn common_prefix(lhs: &ast::Path, rhs: &ast::Path) -> Option<(ast::Path, ast::Path)> {
    let mut res = None;
    let mut lhs_curr = lhs.first_qualifier_or_self();
    let mut rhs_curr = rhs.first_qualifier_or_self();
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

/// Path comparison func for binary searching for merging.
fn path_cmp_bin_search(lhs: Option<ast::Path>, rhs: Option<&ast::Path>) -> Ordering {
    match (lhs.as_ref().and_then(ast::Path::first_segment), rhs.and_then(ast::Path::first_segment))
    {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(ref a), Some(ref b)) => path_segment_cmp(a, b),
    }
}

/// Orders use trees following `rustfmt`'s algorithm for ordering imports, which is `self`, `super` and `crate` first,
/// then identifier imports with lowercase ones first and upper snake case (e.g. UPPER_SNAKE_CASE) ones last,
/// then glob imports, and at last list imports.
///
/// Example foo::{self, foo, baz, Baz, Qux, FOO_BAZ, *, {Bar}}
/// Ref: <https://github.com/rust-lang/rustfmt/blob/6356fca675bd756d71f5c123cd053d17b16c573e/src/imports.rs#L83-L86>.
pub(super) fn use_tree_cmp(a: &ast::UseTree, b: &ast::UseTree) -> Ordering {
    let tiebreak_by_glob_or_alias = || match (a.star_token().is_some(), b.star_token().is_some()) {
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => match (a.rename(), b.rename()) {
            (None, None) => Ordering::Equal,
            (Some(_), None) => Ordering::Greater,
            (None, Some(_)) => Ordering::Less,
            (Some(a_rename), Some(b_rename)) => a_rename
                .name()
                .as_ref()
                .map(ast::Name::text)
                .as_ref()
                .map_or("_", TokenText::as_str)
                .cmp(
                    b_rename
                        .name()
                        .as_ref()
                        .map(ast::Name::text)
                        .as_ref()
                        .map_or("_", TokenText::as_str),
                ),
        },
    };
    let tiebreak_by_tree_list_glob_or_alias = || match (a.use_tree_list(), b.use_tree_list()) {
        (None, None) => tiebreak_by_glob_or_alias(),
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (Some(a_list), Some(b_list)) => a_list
            .use_trees()
            .zip_longest(b_list.use_trees())
            .find_map(|zipped| match zipped {
                EitherOrBoth::Both(ref a_tree, ref b_tree) => match use_tree_cmp(a_tree, b_tree) {
                    Ordering::Equal => None,
                    ord => Some(ord),
                },
                EitherOrBoth::Left(_) => Some(Ordering::Greater),
                EitherOrBoth::Right(_) => Some(Ordering::Less),
            })
            .unwrap_or_else(tiebreak_by_glob_or_alias),
    };
    match (a.path(), b.path()) {
        (None, None) => match (a.is_simple_path(), b.is_simple_path()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => tiebreak_by_tree_list_glob_or_alias(),
        },
        (Some(_), None) if !b.is_simple_path() => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) if !a.is_simple_path() => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (Some(ref a_path), Some(ref b_path)) => {
            // cmp_by would be useful for us here but that is currently unstable
            // cmp doesn't work due the lifetimes on text's return type
            a_path
                .segments()
                .zip_longest(b_path.segments())
                .find_map(|zipped| match zipped {
                    EitherOrBoth::Both(ref a_segment, ref b_segment) => {
                        match path_segment_cmp(a_segment, b_segment) {
                            Ordering::Equal => None,
                            ord => Some(ord),
                        }
                    }
                    EitherOrBoth::Left(_) if b.is_simple_path() => Some(Ordering::Greater),
                    EitherOrBoth::Left(_) => Some(Ordering::Less),
                    EitherOrBoth::Right(_) if a.is_simple_path() => Some(Ordering::Less),
                    EitherOrBoth::Right(_) => Some(Ordering::Greater),
                })
                .unwrap_or_else(tiebreak_by_tree_list_glob_or_alias)
        }
    }
}

fn path_segment_cmp(a: &ast::PathSegment, b: &ast::PathSegment) -> Ordering {
    match (a.kind(), b.kind()) {
        (None, None) => Ordering::Equal,
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        // self
        (Some(PathSegmentKind::SelfKw), Some(PathSegmentKind::SelfKw)) => Ordering::Equal,
        (Some(PathSegmentKind::SelfKw), _) => Ordering::Less,
        (_, Some(PathSegmentKind::SelfKw)) => Ordering::Greater,
        // super
        (Some(PathSegmentKind::SuperKw), Some(PathSegmentKind::SuperKw)) => Ordering::Equal,
        (Some(PathSegmentKind::SuperKw), _) => Ordering::Less,
        (_, Some(PathSegmentKind::SuperKw)) => Ordering::Greater,
        // crate
        (Some(PathSegmentKind::CrateKw), Some(PathSegmentKind::CrateKw)) => Ordering::Equal,
        (Some(PathSegmentKind::CrateKw), _) => Ordering::Less,
        (_, Some(PathSegmentKind::CrateKw)) => Ordering::Greater,
        // identifiers (everything else is treated as an identifier).
        _ => {
            match (
                a.name_ref().as_ref().map(ast::NameRef::text),
                b.name_ref().as_ref().map(ast::NameRef::text),
            ) {
                (None, None) => Ordering::Equal,
                (Some(_), None) => Ordering::Greater,
                (None, Some(_)) => Ordering::Less,
                (Some(a_name), Some(b_name)) => {
                    // snake_case < CamelCase < UPPER_SNAKE_CASE
                    let a_text = a_name.as_str();
                    let b_text = b_name.as_str();
                    fn is_upper_snake_case(s: &str) -> bool {
                        s.chars().all(|c| c.is_uppercase() || c == '_' || c.is_numeric())
                    }
                    if a_text.starts_with(char::is_lowercase)
                        && b_text.starts_with(char::is_uppercase)
                    {
                        return Ordering::Less;
                    }
                    if a_text.starts_with(char::is_uppercase)
                        && b_text.starts_with(char::is_lowercase)
                    {
                        return Ordering::Greater;
                    }
                    if !is_upper_snake_case(a_text) && is_upper_snake_case(b_text) {
                        return Ordering::Less;
                    }
                    if is_upper_snake_case(a_text) && !is_upper_snake_case(b_text) {
                        return Ordering::Greater;
                    }
                    a_text.cmp(&b_text)
                }
            }
        }
    }
}

pub fn eq_visibility(vis0: Option<ast::Visibility>, vis1: Option<ast::Visibility>) -> bool {
    match (vis0, vis1) {
        (None, None) => true,
        (Some(vis0), Some(vis1)) => vis_eq(&vis0, &vis1),
        _ => false,
    }
}

pub fn eq_attrs(
    attrs0: impl Iterator<Item = ast::Attr>,
    attrs1: impl Iterator<Item = ast::Attr>,
) -> bool {
    // FIXME order of attributes should not matter
    let attrs0 = attrs0
        .flat_map(|attr| attr.syntax().descendants_with_tokens())
        .flat_map(|it| it.into_token());
    let attrs1 = attrs1
        .flat_map(|attr| attr.syntax().descendants_with_tokens())
        .flat_map(|it| it.into_token());
    stdx::iter_eq_by(attrs0, attrs1, |tok, tok2| tok.text() == tok2.text())
}

fn path_is_self(path: &ast::Path) -> bool {
    path.segment().and_then(|seg| seg.self_token()).is_some() && path.qualifier().is_none()
}

fn path_len(path: ast::Path) -> usize {
    path.segments().count()
}
