//! Handle syntactic aspects of merging UseTrees.
use std::cmp::Ordering;
use std::iter::empty;

use itertools::{EitherOrBoth, Itertools};
use parser::T;
use stdx::is_upper_snake_case;
use syntax::{
    algo,
    ast::{self, make, AstNode, HasAttrs, HasName, HasVisibility, PathSegmentKind},
    ted::{self, Position},
    Direction,
};

use crate::syntax_helpers::node_ext::vis_eq;

/// What type of merges are allowed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MergeBehavior {
    /// Merge imports from the same crate into a single use statement.
    Crate,
    /// Merge imports from the same module into a single use statement.
    Module,
    /// Merge all imports into a single use statement as long as they have the same visibility
    /// and attributes.
    One,
}

impl MergeBehavior {
    fn is_tree_allowed(&self, tree: &ast::UseTree) -> bool {
        match self {
            MergeBehavior::Crate | MergeBehavior::One => true,
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
    if merge == MergeBehavior::One {
        lhs.wrap_in_tree_list();
        rhs.wrap_in_tree_list();
    } else {
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
    // Sorts the use trees similar to rustfmt's algorithm for ordering imports
    // (see `use_tree_cmp` doc).
    use_trees.sort_unstable_by(use_tree_cmp);
    for rhs_t in rhs.use_tree_list().into_iter().flat_map(|list| list.use_trees()) {
        if !merge.is_tree_allowed(&rhs_t) {
            return None;
        }

        match use_trees.binary_search_by(|lhs_t| use_tree_cmp_bin_search(lhs_t, &rhs_t)) {
            Ok(idx) => {
                let lhs_t = &mut use_trees[idx];
                let lhs_path = lhs_t.path()?;
                let rhs_path = rhs_t.path()?;
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
            Err(insert_idx) => {
                use_trees.insert(insert_idx, rhs_t.clone());
                match lhs.use_tree_list() {
                    // Creates a new use tree list with the item.
                    None => lhs.get_or_create_use_tree_list().add_use_tree(rhs_t),
                    // Recreates the use tree list with sorted items (see `use_tree_cmp` doc).
                    Some(use_tree_list) => {
                        if use_tree_list.l_curly_token().is_none() {
                            ted::insert_raw(
                                Position::first_child_of(use_tree_list.syntax()),
                                make::token(T!['{']),
                            );
                        }
                        if use_tree_list.r_curly_token().is_none() {
                            ted::insert_raw(
                                Position::last_child_of(use_tree_list.syntax()),
                                make::token(T!['}']),
                            );
                        }

                        let mut elements = Vec::new();
                        for (idx, tree) in use_trees.iter().enumerate() {
                            if idx > 0 {
                                elements.push(make::token(T![,]).into());
                                elements.push(make::tokens::single_space().into());
                            }
                            elements.push(tree.syntax().clone().into());
                        }

                        let start = use_tree_list
                            .l_curly_token()
                            .and_then(|l_curly| {
                                algo::non_trivia_sibling(l_curly.into(), Direction::Next)
                            })
                            .filter(|it| it.kind() != T!['}']);
                        let end = use_tree_list
                            .r_curly_token()
                            .and_then(|r_curly| {
                                algo::non_trivia_sibling(r_curly.into(), Direction::Prev)
                            })
                            .filter(|it| it.kind() != T!['{']);
                        if let Some((start, end)) = start.zip(end) {
                            // Attempt to insert elements while preserving preceding and trailing trivia.
                            ted::replace_all(start..=end, elements);
                        } else {
                            let new_use_tree_list = make::use_tree_list(empty()).clone_for_update();
                            let trees_pos = match new_use_tree_list.l_curly_token() {
                                Some(l_curly) => Position::after(l_curly),
                                None => Position::last_child_of(new_use_tree_list.syntax()),
                            };
                            ted::insert_all_raw(trees_pos, elements);
                            ted::replace(use_tree_list.syntax(), new_use_tree_list.syntax());
                        }
                    }
                }
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

/// Use tree comparison func for binary searching for merging.
fn use_tree_cmp_bin_search(lhs: &ast::UseTree, rhs: &ast::UseTree) -> Ordering {
    let lhs_is_simple_path = lhs.is_simple_path() && lhs.rename().is_none();
    let rhs_is_simple_path = rhs.is_simple_path() && rhs.rename().is_none();
    match (
        lhs.path().as_ref().and_then(ast::Path::first_segment),
        rhs.path().as_ref().and_then(ast::Path::first_segment),
    ) {
        (None, None) => match (lhs_is_simple_path, rhs_is_simple_path) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => use_tree_cmp_by_tree_list_glob_or_alias(lhs, rhs, false),
        },
        (Some(_), None) if !rhs_is_simple_path => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) if !lhs_is_simple_path => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (Some(a), Some(b)) => path_segment_cmp(&a, &b),
    }
}

/// Orders use trees following `rustfmt`'s algorithm for ordering imports, which is `self`, `super`
/// and `crate` first, then identifier imports with lowercase ones first and upper snake case
/// (e.g. UPPER_SNAKE_CASE) ones last, then glob imports, and at last list imports.
///
/// Example foo::{self, foo, baz, Baz, Qux, FOO_BAZ, *, {Bar}}
/// Ref: <https://github.com/rust-lang/rustfmt/blob/6356fca675bd756d71f5c123cd053d17b16c573e/src/imports.rs#L83-L86>.
pub(super) fn use_tree_cmp(a: &ast::UseTree, b: &ast::UseTree) -> Ordering {
    let a_is_simple_path = a.is_simple_path() && a.rename().is_none();
    let b_is_simple_path = b.is_simple_path() && b.rename().is_none();
    match (a.path(), b.path()) {
        (None, None) => match (a_is_simple_path, b_is_simple_path) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => use_tree_cmp_by_tree_list_glob_or_alias(a, b, true),
        },
        (Some(_), None) if !b_is_simple_path => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) if !a_is_simple_path => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (Some(a_path), Some(b_path)) => {
            // cmp_by would be useful for us here but that is currently unstable
            // cmp doesn't work due the lifetimes on text's return type
            a_path
                .segments()
                .zip_longest(b_path.segments())
                .find_map(|zipped| match zipped {
                    EitherOrBoth::Both(a_segment, b_segment) => {
                        match path_segment_cmp(&a_segment, &b_segment) {
                            Ordering::Equal => None,
                            ord => Some(ord),
                        }
                    }
                    EitherOrBoth::Left(_) if b_is_simple_path => Some(Ordering::Greater),
                    EitherOrBoth::Left(_) => Some(Ordering::Less),
                    EitherOrBoth::Right(_) if a_is_simple_path => Some(Ordering::Less),
                    EitherOrBoth::Right(_) => Some(Ordering::Greater),
                })
                .unwrap_or_else(|| use_tree_cmp_by_tree_list_glob_or_alias(a, b, true))
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
                    let a_text = a_name.as_str().trim_start_matches("r#");
                    let b_text = b_name.as_str().trim_start_matches("r#");
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
                    a_text.cmp(b_text)
                }
            }
        }
    }
}

/// Orders for use trees with equal paths (see `use_tree_cmp` for details about use tree ordering).
///
/// If the `strict` parameter is set to true and both trees have tree lists, the tree lists are
/// ordered by calling `use_tree_cmp` on their "sub-tree" pairs until either the tie is broken
/// or tree list equality is confirmed, otherwise (i.e. if either `strict` is false or at least
/// one of the trees does *not* have tree list), this potentially recursive step is skipped,
/// and only the presence of a glob pattern or an alias is used to determine the ordering.
fn use_tree_cmp_by_tree_list_glob_or_alias(
    a: &ast::UseTree,
    b: &ast::UseTree,
    strict: bool,
) -> Ordering {
    let cmp_by_glob_or_alias = || match (a.star_token().is_some(), b.star_token().is_some()) {
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
                .map_or("_", |a_name| a_name.as_str().trim_start_matches("r#"))
                .cmp(
                    b_rename
                        .name()
                        .as_ref()
                        .map(ast::Name::text)
                        .as_ref()
                        .map_or("_", |b_name| b_name.as_str().trim_start_matches("r#")),
                ),
        },
    };

    match (a.use_tree_list(), b.use_tree_list()) {
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (Some(a_list), Some(b_list)) if strict => a_list
            .use_trees()
            .zip_longest(b_list.use_trees())
            .find_map(|zipped| match zipped {
                EitherOrBoth::Both(a_tree, b_tree) => match use_tree_cmp(&a_tree, &b_tree) {
                    Ordering::Equal => None,
                    ord => Some(ord),
                },
                EitherOrBoth::Left(_) => Some(Ordering::Greater),
                EitherOrBoth::Right(_) => Some(Ordering::Less),
            })
            .unwrap_or_else(cmp_by_glob_or_alias),
        _ => cmp_by_glob_or_alias(),
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
