//! Handle syntactic aspects of merging UseTrees.
use std::cmp::Ordering;

use itertools::{EitherOrBoth, Itertools};
use parser::T;
use stdx::is_upper_snake_case;
use syntax::{
    Direction, SyntaxElement, algo,
    ast::{
        self, AstNode, HasAttrs, HasName, HasVisibility, PathSegmentKind, edit_in_place::Removable,
        make,
    },
    ted::{self, Position},
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

    // Ignore `None` result because normalization should not affect the merge result.
    try_normalize_use_tree_mut(&lhs_tree, merge_behavior.into());

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

    // Ignore `None` result because normalization should not affect the merge result.
    try_normalize_use_tree_mut(&lhs, merge.into());

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
        if lhs.is_simple_path()
            && rhs.is_simple_path()
            && lhs_path == lhs_prefix
            && rhs_path == rhs_prefix
        {
            // we can't merge if the renames are different (`A as a` and `A as b`),
            // and we can safely return here
            let lhs_name = lhs.rename().and_then(|lhs_name| lhs_name.name());
            let rhs_name = rhs.rename().and_then(|rhs_name| rhs_name.name());
            if lhs_name != rhs_name {
                return None;
            }

            ted::replace(lhs.syntax(), rhs.syntax());
            // we can safely return here, in this case `recursive_merge` doesn't do anything
            return Some(());
        } else {
            lhs.split_prefix(&lhs_prefix);
            rhs.split_prefix(&rhs_prefix);
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
                        (Some(true), None) => {
                            remove_subtree_if_only_self(lhs_t);
                            continue;
                        }
                        (None, Some(true)) => {
                            ted::replace(lhs_t.syntax(), rhs_t.syntax());
                            *lhs_t = rhs_t;
                            remove_subtree_if_only_self(lhs_t);
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
                return None;
            }
            Err(insert_idx) => {
                use_trees.insert(insert_idx, rhs_t.clone());
                // We simply add the use tree to the end of tree list. Ordering of use trees
                // and imports is done by the `try_normalize_*` functions. The sorted `use_trees`
                // vec is only used for binary search.
                lhs.get_or_create_use_tree_list().add_use_tree(rhs_t);
            }
        }
    }
    Some(())
}

/// Style to follow when normalizing a use tree.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NormalizationStyle {
    /// Merges all descendant use tree lists with only one child use tree into their parent use tree.
    ///
    /// Examples:
    /// - `foo::{bar::{Qux}}` -> `foo::bar::Qux`
    /// - `foo::{bar::{self}}` -> `foo::bar`
    /// - `{foo::bar}` -> `foo::bar`
    Default,
    /// Same as default but wraps the root use tree in a use tree list.
    ///
    /// Examples:
    /// - `foo::{bar::{Qux}}` -> `{foo::bar::Qux}`
    /// - `foo::{bar::{self}}` -> `{foo::bar}`
    /// - `{foo::bar}` -> `{foo::bar}`
    One,
}

impl From<MergeBehavior> for NormalizationStyle {
    fn from(mb: MergeBehavior) -> Self {
        match mb {
            MergeBehavior::One => NormalizationStyle::One,
            _ => NormalizationStyle::Default,
        }
    }
}

/// Normalizes a use item by:
/// - Ordering all use trees
/// - Merging use trees with common prefixes
/// - Removing redundant braces based on the specified normalization style
///   (see [`NormalizationStyle`] doc)
///
/// Examples:
///
/// Using the "Default" normalization style
///
/// - `foo::{bar::Qux, bar::{self}}` -> `foo::bar::{self, Qux}`
/// - `foo::bar::{self}` -> `foo::bar`
/// - `{foo::bar}` -> `foo::bar`
///
/// Using the "One" normalization style
///
/// - `foo::{bar::Qux, bar::{self}}` -> `{foo::bar::{self, Qux}}`
/// - `foo::bar::{self}` -> `{foo::bar}`
/// - `foo::bar` -> `{foo::bar}`
pub fn try_normalize_import(use_item: &ast::Use, style: NormalizationStyle) -> Option<ast::Use> {
    let use_item = use_item.clone_subtree().clone_for_update();
    try_normalize_use_tree_mut(&use_item.use_tree()?, style)?;
    Some(use_item)
}

/// Normalizes a use tree (see [`try_normalize_import`] doc).
pub fn try_normalize_use_tree(
    use_tree: &ast::UseTree,
    style: NormalizationStyle,
) -> Option<ast::UseTree> {
    let use_tree = use_tree.clone_subtree().clone_for_update();
    try_normalize_use_tree_mut(&use_tree, style)?;
    Some(use_tree)
}

pub fn try_normalize_use_tree_mut(
    use_tree: &ast::UseTree,
    style: NormalizationStyle,
) -> Option<()> {
    if style == NormalizationStyle::One {
        let mut modified = false;
        modified |= use_tree.wrap_in_tree_list().is_some();
        modified |= recursive_normalize(use_tree, style).is_some();
        if !modified {
            // Either the use tree was already normalized or its semantically empty.
            return None;
        }
    } else {
        recursive_normalize(use_tree, NormalizationStyle::Default)?;
    }
    Some(())
}

/// Recursively normalizes a use tree and its subtrees (if any).
fn recursive_normalize(use_tree: &ast::UseTree, style: NormalizationStyle) -> Option<()> {
    let use_tree_list = use_tree.use_tree_list()?;
    let merge_subtree_into_parent_tree = |single_subtree: &ast::UseTree| {
        let subtree_is_only_self = single_subtree.path().as_ref().is_some_and(path_is_self);

        let merged_path = match (use_tree.path(), single_subtree.path()) {
            // If the subtree is `{self}` then we cannot merge: `use
            // foo::bar::{self}` is not equivalent to `use foo::bar`. See
            // https://github.com/rust-lang/rust-analyzer/pull/17140#issuecomment-2079189725.
            _ if subtree_is_only_self => None,

            (None, None) => None,
            (Some(outer), None) => Some(outer),
            (None, Some(inner)) => Some(inner),
            (Some(outer), Some(inner)) => Some(make::path_concat(outer, inner).clone_for_update()),
        };

        if merged_path.is_some()
            || single_subtree.use_tree_list().is_some()
            || single_subtree.star_token().is_some()
        {
            ted::remove_all_iter(use_tree.syntax().children_with_tokens());
            if let Some(path) = merged_path {
                ted::insert_raw(Position::first_child_of(use_tree.syntax()), path.syntax());
                if single_subtree.use_tree_list().is_some() || single_subtree.star_token().is_some()
                {
                    ted::insert_raw(
                        Position::last_child_of(use_tree.syntax()),
                        make::token(T![::]),
                    );
                }
            }
            if let Some(inner_use_tree_list) = single_subtree.use_tree_list() {
                ted::insert_raw(
                    Position::last_child_of(use_tree.syntax()),
                    inner_use_tree_list.syntax(),
                );
            } else if single_subtree.star_token().is_some() {
                ted::insert_raw(Position::last_child_of(use_tree.syntax()), make::token(T![*]));
            } else if let Some(rename) = single_subtree.rename() {
                ted::insert_raw(
                    Position::last_child_of(use_tree.syntax()),
                    make::tokens::single_space(),
                );
                ted::insert_raw(Position::last_child_of(use_tree.syntax()), rename.syntax());
            }
            Some(())
        } else {
            // Bail on semantically empty use trees.
            None
        }
    };
    let one_style_tree_list = |subtree: &ast::UseTree| match (
        subtree.path().is_none() && subtree.star_token().is_none() && subtree.rename().is_none(),
        subtree.use_tree_list(),
    ) {
        (true, tree_list) => tree_list,
        _ => None,
    };
    let add_element_to_list = |elem: SyntaxElement, elements: &mut Vec<SyntaxElement>| {
        if !elements.is_empty() {
            elements.push(make::token(T![,]).into());
            elements.push(make::tokens::single_space().into());
        }
        elements.push(elem);
    };
    if let Some((single_subtree,)) = use_tree_list.use_trees().collect_tuple() {
        if style == NormalizationStyle::One {
            // Only normalize descendant subtrees if the normalization style is "one".
            recursive_normalize(&single_subtree, NormalizationStyle::Default)?;
        } else {
            // Otherwise, merge the single subtree into it's parent (if possible)
            // and then normalize the result.
            merge_subtree_into_parent_tree(&single_subtree)?;
            recursive_normalize(use_tree, style);
        }
    } else {
        // Tracks whether any changes have been made to the use tree.
        let mut modified = false;

        // Recursively un-nests (if necessary) and then normalizes each subtree in the tree list.
        for subtree in use_tree_list.use_trees() {
            if let Some(one_tree_list) = one_style_tree_list(&subtree) {
                let mut elements = Vec::new();
                let mut one_tree_list_iter = one_tree_list.use_trees();
                let mut prev_skipped = Vec::new();
                loop {
                    let mut prev_skipped_iter = prev_skipped.into_iter();
                    let mut curr_skipped = Vec::new();

                    while let Some(sub_sub_tree) =
                        one_tree_list_iter.next().or(prev_skipped_iter.next())
                    {
                        if let Some(sub_one_tree_list) = one_style_tree_list(&sub_sub_tree) {
                            curr_skipped.extend(sub_one_tree_list.use_trees());
                        } else {
                            modified |=
                                recursive_normalize(&sub_sub_tree, NormalizationStyle::Default)
                                    .is_some();
                            add_element_to_list(
                                sub_sub_tree.syntax().clone().into(),
                                &mut elements,
                            );
                        }
                    }

                    if curr_skipped.is_empty() {
                        // Un-nesting is complete.
                        break;
                    }
                    prev_skipped = curr_skipped;
                }

                // Either removes the subtree (if its semantically empty) or replaces it with
                // the un-nested elements.
                if elements.is_empty() {
                    subtree.remove();
                } else {
                    ted::replace_with_many(subtree.syntax(), elements);
                }
                modified = true;
            } else {
                modified |= recursive_normalize(&subtree, NormalizationStyle::Default).is_some();
            }
        }

        // Merge all merge-able subtrees.
        let mut tree_list_iter = use_tree_list.use_trees();
        let mut anchor = tree_list_iter.next()?;
        let mut prev_skipped = Vec::new();
        loop {
            let mut has_merged = false;
            let mut prev_skipped_iter = prev_skipped.into_iter();
            let mut next_anchor = None;
            let mut curr_skipped = Vec::new();

            while let Some(candidate) = tree_list_iter.next().or(prev_skipped_iter.next()) {
                let result = try_merge_trees_mut(&anchor, &candidate, MergeBehavior::Crate);
                if result.is_some() {
                    // Remove merged subtree.
                    candidate.remove();
                    has_merged = true;
                } else if next_anchor.is_none() {
                    next_anchor = Some(candidate);
                } else {
                    curr_skipped.push(candidate);
                }
            }

            if has_merged {
                // Normalize the merge result.
                recursive_normalize(&anchor, NormalizationStyle::Default);
                modified = true;
            }

            let (Some(next_anchor), true) = (next_anchor, !curr_skipped.is_empty()) else {
                // Merging is complete.
                break;
            };

            // Try to merge the remaining subtrees in the next iteration.
            anchor = next_anchor;
            prev_skipped = curr_skipped;
        }

        let mut subtrees: Vec<_> = use_tree_list.use_trees().collect();
        // Merge the remaining subtree into its parent, if its only one and
        // the normalization style is not "one".
        if subtrees.len() == 1 && style != NormalizationStyle::One {
            modified |= merge_subtree_into_parent_tree(&subtrees[0]).is_some();
        }
        // Order the remaining subtrees (if necessary).
        if subtrees.len() > 1 {
            let mut did_sort = false;
            subtrees.sort_unstable_by(|a, b| {
                let order = use_tree_cmp_bin_search(a, b);
                if !did_sort && order == Ordering::Less {
                    did_sort = true;
                }
                order
            });
            if did_sort {
                let start = use_tree_list
                    .l_curly_token()
                    .and_then(|l_curly| algo::non_trivia_sibling(l_curly.into(), Direction::Next))
                    .filter(|it| it.kind() != T!['}']);
                let end = use_tree_list
                    .r_curly_token()
                    .and_then(|r_curly| algo::non_trivia_sibling(r_curly.into(), Direction::Prev))
                    .filter(|it| it.kind() != T!['{']);
                if let Some((start, end)) = start.zip(end) {
                    // Attempt to insert elements while preserving preceding and trailing trivia.
                    let mut elements = Vec::new();
                    for subtree in subtrees {
                        add_element_to_list(subtree.syntax().clone().into(), &mut elements);
                    }
                    ted::replace_all(start..=end, elements);
                } else {
                    let new_use_tree_list = make::use_tree_list(subtrees).clone_for_update();
                    ted::replace(use_tree_list.syntax(), new_use_tree_list.syntax());
                }
                modified = true;
            }
        }

        if !modified {
            // Either the use tree was already normalized or its semantically empty.
            return None;
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
/// Example: `foo::{self, baz, foo, Baz, Qux, FOO_BAZ, *, {Bar}}`
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
                    // snake_case < UpperCamelCase < UPPER_SNAKE_CASE
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

fn get_single_subtree(use_tree: &ast::UseTree) -> Option<ast::UseTree> {
    use_tree
        .use_tree_list()
        .and_then(|tree_list| tree_list.use_trees().collect_tuple())
        .map(|(single_subtree,)| single_subtree)
}

fn remove_subtree_if_only_self(use_tree: &ast::UseTree) {
    let Some(single_subtree) = get_single_subtree(use_tree) else { return };
    match (use_tree.path(), single_subtree.path()) {
        (Some(_), Some(inner)) if path_is_self(&inner) => {
            ted::remove_all_iter(single_subtree.syntax().children_with_tokens());
        }
        _ => (),
    }
}
