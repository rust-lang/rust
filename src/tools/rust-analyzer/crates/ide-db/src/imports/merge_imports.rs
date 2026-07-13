//! Handle syntactic aspects of merging UseTrees.
use std::cmp::Ordering;

use itertools::{EitherOrBoth, Itertools};
use parser::T;
use syntax::{
    ToSmolStr,
    ast::{
        self, AstNode, HasAttrs, HasName, HasVisibility, PathSegmentKind,
        syntax_factory::SyntaxFactory,
    },
    syntax_editor::{Position, SyntaxEditor},
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
pub fn try_merge_imports(
    make: &SyntaxFactory,
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

    let lhs_tree = lhs.use_tree()?;
    let rhs_tree = rhs.use_tree()?;
    let merged_tree = try_merge_trees_with_factory(lhs_tree, rhs_tree, merge_behavior, make)?;

    // Ignore `None` result because normalization should not affect the merge result.
    let use_tree = try_normalize_use_tree(merged_tree.clone(), merge_behavior.into(), make)
        .unwrap_or(merged_tree);

    make_use_with_tree(lhs, use_tree)
}

/// Merge `rhs` into `lhs` keeping both intact.
pub fn try_merge_trees(
    make: &SyntaxFactory,
    lhs: &ast::UseTree,
    rhs: &ast::UseTree,
    merge: MergeBehavior,
) -> Option<ast::UseTree> {
    let merged = try_merge_trees_with_factory(lhs.clone(), rhs.clone(), merge, make)?;

    // Ignore `None` result because normalization should not affect the merge result.
    Some(try_normalize_use_tree(merged.clone(), merge.into(), make).unwrap_or(merged))
}

fn try_merge_trees_with_factory(
    mut lhs: ast::UseTree,
    mut rhs: ast::UseTree,
    merge: MergeBehavior,
    make: &SyntaxFactory,
) -> Option<ast::UseTree> {
    if merge == MergeBehavior::One {
        lhs = wrap_in_tree_list(&lhs, make).unwrap_or(lhs);
        rhs = wrap_in_tree_list(&rhs, make).unwrap_or(rhs);
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
            if lhs_name.as_ref().map(|name| name.text())
                != rhs_name.as_ref().map(|name| name.text())
            {
                return None;
            }

            return Some(rhs);
        } else {
            lhs = split_prefix(&lhs, &lhs_prefix, make)?;
            rhs = split_prefix(&rhs, &rhs_prefix, make)?;
        }
    }
    recursive_merge(lhs, rhs, merge, make)
}

/// Recursively merges rhs to lhs
#[must_use]
fn recursive_merge(
    lhs: ast::UseTree,
    rhs: ast::UseTree,
    merge: MergeBehavior,
    make: &SyntaxFactory,
) -> Option<ast::UseTree> {
    let mut use_trees: Vec<ast::UseTree> = lhs
        .use_tree_list()?
        .use_trees()
        // We use Option here to early return from this function. This is not the
        // same as a `filter` op.
        .map(|tree| merge.is_tree_allowed(&tree).then_some(tree))
        .collect::<Option<_>>()?;

    // Sorts the use trees similar to rustfmt's algorithm for ordering imports
    // (see `use_tree_cmp` doc).
    use_trees.sort_unstable_by(use_tree_cmp);

    for rhs_t in rhs.use_tree_list()?.use_trees() {
        if !merge.is_tree_allowed(&rhs_t) {
            return None;
        }

        match use_trees.binary_search_by(|lhs_t| use_tree_cmp_bin_search(lhs_t, &rhs_t)) {
            Ok(idx) => {
                let mut lhs_t = use_trees[idx].clone();
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
                        use_trees[idx] = rhs_t;
                        continue;
                    }

                    match (tree_contains_self(&lhs_t), tree_contains_self(&rhs_t)) {
                        (Some(true), None) => {
                            lhs_t = remove_subtree_if_only_self(lhs_t, make)?;
                            use_trees[idx] = lhs_t;
                            continue;
                        }
                        (None, Some(true)) => {
                            lhs_t = rhs_t;
                            lhs_t = remove_subtree_if_only_self(lhs_t, make)?;
                            use_trees[idx] = lhs_t;
                            continue;
                        }
                        _ => (),
                    }

                    if lhs_t.is_simple_path() && rhs_t.is_simple_path() {
                        continue;
                    }
                }

                lhs_t = split_prefix(&lhs_t, &lhs_prefix, make)?;
                let rhs_t = split_prefix(&rhs_t, &rhs_prefix, make)?;
                lhs_t = recursive_merge(lhs_t, rhs_t, merge, make)?;
                use_trees[idx] = lhs_t;
            }
            Err(_)
                if merge == MergeBehavior::Module
                    && !use_trees.is_empty()
                    && rhs_t.use_tree_list().is_some() =>
            {
                return None;
            }
            Err(insert_idx) => {
                use_trees.insert(insert_idx, rhs_t);
            }
        }
    }

    with_use_tree_list(&lhs, use_trees, make)
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
pub fn try_normalize_import(
    make: &SyntaxFactory,
    use_item: &ast::Use,
    style: NormalizationStyle,
) -> Option<ast::Use> {
    let use_tree = try_normalize_use_tree(use_item.use_tree()?, style, make)?;

    make_use_with_tree(use_item, use_tree)
}

fn try_normalize_use_tree(
    use_tree: ast::UseTree,
    style: NormalizationStyle,
    make: &SyntaxFactory,
) -> Option<ast::UseTree> {
    if style == NormalizationStyle::One {
        let mut use_tree = use_tree;
        let mut modified = false;
        if let Some(wrapped) = wrap_in_tree_list(&use_tree, make) {
            use_tree = wrapped;
            modified = true;
        }
        if let Some(normalized) = recursive_normalize(use_tree.clone(), style, make) {
            use_tree = normalized;
            modified = true;
        }
        return modified.then_some(use_tree);
    }

    recursive_normalize(use_tree, NormalizationStyle::Default, make)
}

/// Recursively normalizes a use tree and its subtrees (if any).
fn recursive_normalize(
    use_tree: ast::UseTree,
    style: NormalizationStyle,
    make: &SyntaxFactory,
) -> Option<ast::UseTree> {
    let use_tree_list = use_tree.use_tree_list()?;
    let mut subtrees = use_tree_list.use_trees().collect::<Vec<_>>();
    if subtrees.len() == 1 {
        if style == NormalizationStyle::One {
            let subtree = subtrees.pop()?;
            let normalized = recursive_normalize(subtree, NormalizationStyle::Default, make)?;
            return with_use_tree_list(&use_tree, vec![normalized], make);
        }

        let merged = merge_single_subtree_into_parent_tree(use_tree, make)?;
        return Some(recursive_normalize(merged.clone(), style, make).unwrap_or(merged));
    }

    let mut modified = false;
    let mut new_use_tree_list = Vec::new();
    for subtree in subtrees {
        if one_style_tree_list(&subtree).is_some() {
            let mut elements = Vec::new();
            flatten_one_style_tree(subtree, &mut elements, &mut modified, make);
            new_use_tree_list.extend(elements);
            modified = true;
        } else if let Some(normalized) =
            recursive_normalize(subtree.clone(), NormalizationStyle::Default, make)
        {
            new_use_tree_list.push(normalized);
            modified = true;
        } else {
            new_use_tree_list.push(subtree);
        }
    }

    let mut use_tree =
        if modified { with_use_tree_list(&use_tree, new_use_tree_list, make)? } else { use_tree };

    let mut use_tree_list = use_tree.use_tree_list()?.use_trees().collect::<Vec<_>>();
    let mut anchor_idx = 0;
    let mut merged_any = false;
    while anchor_idx < use_tree_list.len() {
        let mut candidate_idx = anchor_idx + 1;
        while candidate_idx < use_tree_list.len() {
            if let Some(mut merged) = try_merge_trees_with_factory(
                use_tree_list[anchor_idx].clone(),
                use_tree_list[candidate_idx].clone(),
                MergeBehavior::Crate,
                make,
            ) {
                if let Some(normalized) =
                    recursive_normalize(merged.clone(), NormalizationStyle::Default, make)
                {
                    merged = normalized;
                }

                use_tree_list[anchor_idx] = merged;
                use_tree_list.remove(candidate_idx);
                merged_any = true;
            } else {
                candidate_idx += 1;
            }
        }

        anchor_idx += 1;
    }
    if merged_any {
        use_tree = with_use_tree_list(&use_tree, use_tree_list, make)?;
        modified = true;
    }

    if style != NormalizationStyle::One {
        let subtrees = use_tree.use_tree_list()?.use_trees().collect::<Vec<_>>();
        if subtrees.len() == 1
            && let Some(merged) = merge_single_subtree_into_parent_tree(use_tree.clone(), make)
        {
            use_tree = merged;
            modified = true;
        }
    }

    if let Some(list) = use_tree.use_tree_list() {
        let mut use_tree_list = list.use_trees().collect::<Vec<_>>();
        if use_tree_list
            .windows(2)
            .any(|trees| use_tree_cmp_bin_search(&trees[0], &trees[1]).is_gt())
        {
            use_tree_list.sort_unstable_by(use_tree_cmp_bin_search);
            use_tree = with_use_tree_list(&use_tree, use_tree_list, make)?;
            modified = true;
        }
    }

    modified.then_some(use_tree)
}

fn flatten_one_style_tree(
    subtree: ast::UseTree,
    elements: &mut Vec<ast::UseTree>,
    modified: &mut bool,
    make: &SyntaxFactory,
) {
    let Some(one_tree_list) = one_style_tree_list(&subtree) else { return };
    let mut one_tree_list_iter = one_tree_list.use_trees();
    let mut prev_skipped = Vec::new();
    loop {
        let mut prev_skipped_iter = prev_skipped.into_iter();
        let mut curr_skipped = Vec::new();

        while let Some(sub_sub_tree) =
            one_tree_list_iter.next().or_else(|| prev_skipped_iter.next())
        {
            if let Some(sub_one_tree_list) = one_style_tree_list(&sub_sub_tree) {
                curr_skipped.extend(sub_one_tree_list.use_trees());
            } else if let Some(normalized) =
                recursive_normalize(sub_sub_tree.clone(), NormalizationStyle::Default, make)
            {
                *modified = true;
                elements.push(normalized);
            } else {
                elements.push(sub_sub_tree);
            }
        }

        if curr_skipped.is_empty() {
            break;
        }
        prev_skipped = curr_skipped;
    }
}

fn merge_single_subtree_into_parent_tree(
    use_tree: ast::UseTree,
    make: &SyntaxFactory,
) -> Option<ast::UseTree> {
    let single_subtree = get_single_subtree(&use_tree)?;
    let subtree_is_only_self = single_subtree.path().as_ref().is_some_and(path_is_self);

    let merged_path = match (use_tree.path(), single_subtree.path()) {
        _ if subtree_is_only_self => None,
        (None, None) => None,
        (Some(outer), None) => Some(outer),
        (None, Some(inner)) => Some(inner),
        (Some(outer), Some(inner)) => Some(make.path_concat(outer, inner)),
    };

    let list = single_subtree.use_tree_list();
    let list_is_none = list.is_none();
    let star = single_subtree.star_token().is_some();
    if merged_path.is_some() || list.is_some() || star {
        let rename = (!star && list_is_none).then(|| single_subtree.rename()).flatten();
        make_use_tree_from_parts(make, merged_path, list, rename, star)
    } else {
        None
    }
}

fn one_style_tree_list(subtree: &ast::UseTree) -> Option<ast::UseTreeList> {
    (subtree.path().is_none() && subtree.star_token().is_none() && subtree.rename().is_none())
        .then(|| subtree.use_tree_list())
        .flatten()
}

fn remove_subtree_if_only_self(
    use_tree: ast::UseTree,
    make: &SyntaxFactory,
) -> Option<ast::UseTree> {
    let Some(single_subtree) = get_single_subtree(&use_tree) else {
        return Some(use_tree);
    };
    match (use_tree.path(), single_subtree.path()) {
        (Some(path), Some(inner)) if path_is_self(&inner) => {
            Some(make.use_tree(path, None, use_tree.rename(), false))
        }
        _ => Some(use_tree),
    }
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
    let lhs_segment = lhs.path().and_then(|path| path.first_segment());
    let rhs_segment = rhs.path().and_then(|path| path.first_segment());
    match (lhs_segment, rhs_segment) {
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

/// Orders use trees following `rustfmt`'s version sorting algorithm for ordering imports.
///
/// Example: `foo::{self, Baz, FOO_BAZ, Qux, baz, foo, *, {Bar}}`
///
/// Ref:
///   - <https://doc.rust-lang.org/style-guide/index.html#sorting>
///   - <https://doc.rust-lang.org/edition-guide/rust-2024/rustfmt.html>
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
                    let a_text = a_name.as_str().trim_start_matches("r#");
                    let b_text = b_name.as_str().trim_start_matches("r#");
                    version_sort::version_sort(a_text, b_text)
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
    let mut attrs0: Vec<_> = attrs0.map(|attr| attr.syntax().text().to_smolstr()).collect();
    let mut attrs1: Vec<_> = attrs1.map(|attr| attr.syntax().text().to_smolstr()).collect();
    attrs0.sort_unstable();
    attrs1.sort_unstable();

    attrs0 == attrs1
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

fn make_use_with_tree(original: &ast::Use, use_tree: ast::UseTree) -> Option<ast::Use> {
    let (editor, use_item) = SyntaxEditor::with_ast_node(original);
    let original_tree = use_item.use_tree()?;
    editor.replace(original_tree.syntax(), use_tree.syntax());
    let edit = editor.finish();
    ast::Use::cast(edit.new_root().clone())
}

fn make_use_tree_list(
    make: &SyntaxFactory,
    use_trees: Vec<ast::UseTree>,
    style_source: Option<&ast::UseTreeList>,
) -> Option<ast::UseTreeList> {
    let use_tree_list = make.use_tree_list(use_trees);
    let Some(style_source) = style_source else {
        return Some(use_tree_list);
    };

    let source_l_curly = style_source.l_curly_token()?;
    let source_r_curly = style_source.r_curly_token()?;

    let leading_ws = source_l_curly.next_token().filter(|token| token.kind().is_trivia());

    let trailing_ws = source_r_curly.prev_token().filter(|token| token.kind().is_trivia());

    let source_trailing_token = trailing_ws
        .as_ref()
        .and_then(|token| token.prev_token())
        .or_else(|| source_r_curly.prev_token());

    let source_has_trailing_comma =
        source_trailing_token.is_some_and(|token| token.kind() == T![,]);

    let (editor, use_tree_list) = SyntaxEditor::with_ast_node(&use_tree_list);
    let make = editor.make();

    if let Some(leading_ws) = leading_ws {
        editor.insert(
            Position::after(use_tree_list.l_curly_token()?),
            make.whitespace(leading_ws.text()),
        );
    }

    let r_curly = use_tree_list.r_curly_token()?;

    let generated_has_trailing_comma = r_curly
        .prev_token()
        .and_then(|token| if token.kind().is_trivia() { token.prev_token() } else { Some(token) })
        .is_some_and(|token| token.kind() == T![,]);

    let mut trailing = Vec::new();

    if source_has_trailing_comma
        && !generated_has_trailing_comma
        && use_tree_list.use_trees().next().is_some()
    {
        trailing.push(make.token(T![,]).into());
    }

    if let Some(trailing_ws) = trailing_ws {
        trailing.push(make.whitespace(trailing_ws.text()).into());
    }

    if !trailing.is_empty() {
        editor.insert_all(Position::before(r_curly), trailing);
    }

    let edit = editor.finish();
    ast::UseTreeList::cast(edit.new_root().clone())
}

fn make_use_tree_from_list(make: &SyntaxFactory, list: ast::UseTreeList) -> Option<ast::UseTree> {
    let placeholder = make.use_tree_glob();
    let (editor, use_tree) = SyntaxEditor::with_ast_node(&placeholder);
    let first_child = use_tree.syntax().first_child_or_token()?;
    let last_child = use_tree.syntax().last_child_or_token()?;
    editor.replace_all(first_child..=last_child, vec![list.syntax().clone().into()]);
    let edit = editor.finish();
    ast::UseTree::cast(edit.new_root().clone())
}

fn make_use_tree_from_parts(
    make: &SyntaxFactory,
    path: Option<ast::Path>,
    list: Option<ast::UseTreeList>,
    rename: Option<ast::Rename>,
    star: bool,
) -> Option<ast::UseTree> {
    match (path, list, star) {
        (Some(path), list, star) => Some(make.use_tree(path, list, rename, star)),
        (None, Some(list), false) if rename.is_none() => make_use_tree_from_list(make, list),
        (None, None, true) if rename.is_none() => Some(make.use_tree_glob()),
        (None, None, false) if rename.is_none() => None,
        _ => None,
    }
}

fn with_use_tree_list(
    use_tree: &ast::UseTree,
    use_trees: Vec<ast::UseTree>,
    make: &SyntaxFactory,
) -> Option<ast::UseTree> {
    let list = make_use_tree_list(make, use_trees, use_tree.use_tree_list().as_ref())?;
    make_use_tree_from_parts(
        make,
        use_tree.path(),
        Some(list),
        use_tree.rename(),
        use_tree.star_token().is_some(),
    )
}

pub(crate) fn wrap_in_tree_list(
    use_tree: &ast::UseTree,
    make: &SyntaxFactory,
) -> Option<ast::UseTree> {
    if use_tree.path().is_none()
        && use_tree.use_tree_list().is_some()
        && use_tree.rename().is_none()
        && use_tree.star_token().is_none()
    {
        return None;
    }

    let list = make_use_tree_list(make, vec![use_tree.clone()], None)?;
    make_use_tree_from_list(make, list)
}

fn split_prefix(
    use_tree: &ast::UseTree,
    prefix: &ast::Path,
    make: &SyntaxFactory,
) -> Option<ast::UseTree> {
    let path = use_tree.path()?;
    if path == *prefix && use_tree.use_tree_list().is_some() {
        return Some(use_tree.clone());
    }

    let suffix = if path == *prefix {
        if use_tree.star_token().is_some() {
            make.use_tree_glob()
        } else {
            let self_path = make.path_unqualified(make.path_segment_self());
            make.use_tree(self_path, None, use_tree.rename(), false)
        }
    } else {
        let suffix_segments: Vec<_> = path.segments().skip(prefix.segments().count()).collect();
        if suffix_segments.is_empty() {
            return None;
        }
        let suffix_path = make.path_from_segments(suffix_segments, false);
        make.use_tree(
            suffix_path,
            use_tree.use_tree_list(),
            use_tree.rename(),
            use_tree.star_token().is_some(),
        )
    };

    let list = make_use_tree_list(make, vec![suffix], None)?;
    Some(make.use_tree(prefix.clone(), Some(list), None, false))
}

// Taken from rustfmt
// https://github.com/rust-lang/rustfmt/blob/0332da01486508710f2a542111e40513bfb215aa/src/sort.rs
mod version_sort {
    // Original rustfmt code contains some clippy lints.
    // Suppress them to minimize changes from upstream.
    #![allow(clippy::all)]

    use std::cmp::Ordering;

    use itertools::{EitherOrBoth, Itertools};

    struct VersionChunkIter<'a> {
        ident: &'a str,
        start: usize,
    }

    impl<'a> VersionChunkIter<'a> {
        pub(crate) fn new(ident: &'a str) -> Self {
            Self { ident, start: 0 }
        }

        fn parse_numeric_chunk(
            &mut self,
            mut chars: std::str::CharIndices<'a>,
        ) -> Option<VersionChunk<'a>> {
            let mut end = self.start;
            let mut is_end_of_chunk = false;

            while let Some((idx, c)) = chars.next() {
                end = self.start + idx;

                if c.is_ascii_digit() {
                    continue;
                }

                is_end_of_chunk = true;
                break;
            }

            let source = if is_end_of_chunk {
                let value = &self.ident[self.start..end];
                self.start = end;
                value
            } else {
                let value = &self.ident[self.start..];
                self.start = self.ident.len();
                value
            };

            let zeros = source.chars().take_while(|c| *c == '0').count();
            let value = source.parse::<usize>().ok()?;

            Some(VersionChunk::Number { value, zeros, source })
        }

        fn parse_str_chunk(
            &mut self,
            mut chars: std::str::CharIndices<'a>,
        ) -> Option<VersionChunk<'a>> {
            let mut end = self.start;
            let mut is_end_of_chunk = false;

            while let Some((idx, c)) = chars.next() {
                end = self.start + idx;

                if c == '_' {
                    is_end_of_chunk = true;
                    break;
                }

                if !c.is_ascii_digit() {
                    continue;
                }

                is_end_of_chunk = true;
                break;
            }

            let source = if is_end_of_chunk {
                let value = &self.ident[self.start..end];
                self.start = end;
                value
            } else {
                let value = &self.ident[self.start..];
                self.start = self.ident.len();
                value
            };

            Some(VersionChunk::Str(source))
        }
    }

    impl<'a> Iterator for VersionChunkIter<'a> {
        type Item = VersionChunk<'a>;

        fn next(&mut self) -> Option<Self::Item> {
            let mut chars = self.ident[self.start..].char_indices();
            let (_, next) = chars.next()?;

            if next == '_' {
                self.start = self.start + next.len_utf8();
                return Some(VersionChunk::Underscore);
            }

            if next.is_ascii_digit() {
                return self.parse_numeric_chunk(chars);
            }

            self.parse_str_chunk(chars)
        }
    }

    /// Represents a chunk in the version-sort algorithm
    #[derive(Debug, PartialEq, Eq)]
    enum VersionChunk<'a> {
        /// A single `_` in an identifier. Underscores are sorted before all other characters.
        Underscore,
        /// A &str chunk in the version sort.
        Str(&'a str),
        /// A numeric chunk in the version sort. Keeps track of the numeric value and leading zeros.
        Number { value: usize, zeros: usize, source: &'a str },
    }

    /// Determine which side of the version-sort comparison had more leading zeros.
    #[derive(Debug, PartialEq, Eq)]
    enum MoreLeadingZeros {
        Left,
        Right,
        Equal,
    }

    pub(super) fn version_sort(a: &str, b: &str) -> Ordering {
        let iter_a = VersionChunkIter::new(a);
        let iter_b = VersionChunkIter::new(b);
        let mut more_leading_zeros = MoreLeadingZeros::Equal;

        for either_or_both in iter_a.zip_longest(iter_b) {
            match either_or_both {
                EitherOrBoth::Left(_) => return std::cmp::Ordering::Greater,
                EitherOrBoth::Right(_) => return std::cmp::Ordering::Less,
                EitherOrBoth::Both(a, b) => match (a, b) {
                    (VersionChunk::Underscore, VersionChunk::Underscore) => {
                        continue;
                    }
                    (VersionChunk::Underscore, _) => return std::cmp::Ordering::Less,
                    (_, VersionChunk::Underscore) => return std::cmp::Ordering::Greater,
                    (VersionChunk::Str(ca), VersionChunk::Str(cb))
                    | (VersionChunk::Str(ca), VersionChunk::Number { source: cb, .. })
                    | (VersionChunk::Number { source: ca, .. }, VersionChunk::Str(cb)) => {
                        match ca.cmp(&cb) {
                            std::cmp::Ordering::Equal => {
                                continue;
                            }
                            order @ _ => return order,
                        }
                    }
                    (
                        VersionChunk::Number { value: va, zeros: lza, .. },
                        VersionChunk::Number { value: vb, zeros: lzb, .. },
                    ) => match va.cmp(&vb) {
                        std::cmp::Ordering::Equal => {
                            if lza == lzb {
                                continue;
                            }

                            if more_leading_zeros == MoreLeadingZeros::Equal && lza > lzb {
                                more_leading_zeros = MoreLeadingZeros::Left;
                            } else if more_leading_zeros == MoreLeadingZeros::Equal && lza < lzb {
                                more_leading_zeros = MoreLeadingZeros::Right;
                            }
                            continue;
                        }
                        order @ _ => return order,
                    },
                },
            }
        }

        match more_leading_zeros {
            MoreLeadingZeros::Equal => std::cmp::Ordering::Equal,
            MoreLeadingZeros::Left => std::cmp::Ordering::Less,
            MoreLeadingZeros::Right => std::cmp::Ordering::Greater,
        }
    }
}
