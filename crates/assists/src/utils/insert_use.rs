//! Handle syntactic aspects of inserting a new `use`.
// FIXME: rewrite according to the plan, outlined in
// https://github.com/rust-analyzer/rust-analyzer/issues/3301#issuecomment-592931553

use std::iter::successors;

use either::Either;
use syntax::{
    ast::{self, NameOwner, VisibilityOwner},
    AstNode, AstToken, Direction, SmolStr,
    SyntaxKind::{PATH, PATH_SEGMENT},
    SyntaxNode, SyntaxToken, T,
};
use text_edit::TextEditBuilder;

use crate::assist_context::AssistContext;

/// Determines the containing syntax node in which to insert a `use` statement affecting `position`.
pub(crate) fn find_insert_use_container(
    position: &SyntaxNode,
    ctx: &AssistContext,
) -> Option<Either<ast::ItemList, ast::SourceFile>> {
    ctx.sema.ancestors_with_macros(position.clone()).find_map(|n| {
        if let Some(module) = ast::Module::cast(n.clone()) {
            return module.item_list().map(|it| Either::Left(it));
        }
        Some(Either::Right(ast::SourceFile::cast(n)?))
    })
}

/// Creates and inserts a use statement for the given path to import.
/// The use statement is inserted in the scope most appropriate to the
/// the cursor position given, additionally merged with the existing use imports.
pub(crate) fn insert_use_statement(
    // Ideally the position of the cursor, used to
    position: &SyntaxNode,
    path_to_import: &str,
    ctx: &AssistContext,
    builder: &mut TextEditBuilder,
) {
    let target = path_to_import.split("::").map(SmolStr::new).collect::<Vec<_>>();
    let container = find_insert_use_container(position, ctx);

    if let Some(container) = container {
        let syntax = container.either(|l| l.syntax().clone(), |r| r.syntax().clone());
        let action = best_action_for_target(syntax, position.clone(), &target);
        make_assist(&action, &target, builder);
    }
}

fn collect_path_segments_raw(
    segments: &mut Vec<ast::PathSegment>,
    mut path: ast::Path,
) -> Option<usize> {
    let oldlen = segments.len();
    loop {
        let mut children = path.syntax().children_with_tokens();
        let (first, second, third) = (
            children.next().map(|n| (n.clone(), n.kind())),
            children.next().map(|n| (n.clone(), n.kind())),
            children.next().map(|n| (n.clone(), n.kind())),
        );
        match (first, second, third) {
            (Some((subpath, PATH)), Some((_, T![::])), Some((segment, PATH_SEGMENT))) => {
                path = ast::Path::cast(subpath.as_node()?.clone())?;
                segments.push(ast::PathSegment::cast(segment.as_node()?.clone())?);
            }
            (Some((segment, PATH_SEGMENT)), _, _) => {
                segments.push(ast::PathSegment::cast(segment.as_node()?.clone())?);
                break;
            }
            (_, _, _) => return None,
        }
    }
    // We need to reverse only the new added segments
    let only_new_segments = segments.split_at_mut(oldlen).1;
    only_new_segments.reverse();
    Some(segments.len() - oldlen)
}

fn fmt_segments_raw(segments: &[SmolStr], buf: &mut String) {
    let mut iter = segments.iter();
    if let Some(s) = iter.next() {
        buf.push_str(s);
    }
    for s in iter {
        buf.push_str("::");
        buf.push_str(s);
    }
}

/// Returns the number of common segments.
fn compare_path_segments(left: &[SmolStr], right: &[ast::PathSegment]) -> usize {
    left.iter().zip(right).take_while(|(l, r)| compare_path_segment(l, r)).count()
}

fn compare_path_segment(a: &SmolStr, b: &ast::PathSegment) -> bool {
    if let Some(kb) = b.kind() {
        match kb {
            ast::PathSegmentKind::Name(nameref_b) => a == nameref_b.text(),
            ast::PathSegmentKind::SelfKw => a == "self",
            ast::PathSegmentKind::SuperKw => a == "super",
            ast::PathSegmentKind::CrateKw => a == "crate",
            ast::PathSegmentKind::Type { .. } => false, // not allowed in imports
        }
    } else {
        false
    }
}

fn compare_path_segment_with_name(a: &SmolStr, b: &ast::Name) -> bool {
    a == b.text()
}

#[derive(Clone, Debug)]
enum ImportAction {
    Nothing,
    // Add a brand new use statement.
    AddNewUse {
        anchor: Option<SyntaxNode>, // anchor node
        add_after_anchor: bool,
    },

    // To split an existing use statement creating a nested import.
    AddNestedImport {
        // how may segments matched with the target path
        common_segments: usize,
        path_to_split: ast::Path,
        // the first segment of path_to_split we want to add into the new nested list
        first_segment_to_split: Option<ast::PathSegment>,
        // Wether to add 'self' in addition to the target path
        add_self: bool,
    },
    // To add the target path to an existing nested import tree list.
    AddInTreeList {
        common_segments: usize,
        // The UseTreeList where to add the target path
        tree_list: ast::UseTreeList,
        add_self: bool,
    },
}

impl ImportAction {
    fn add_new_use(anchor: Option<SyntaxNode>, add_after_anchor: bool) -> Self {
        ImportAction::AddNewUse { anchor, add_after_anchor }
    }

    fn add_nested_import(
        common_segments: usize,
        path_to_split: ast::Path,
        first_segment_to_split: Option<ast::PathSegment>,
        add_self: bool,
    ) -> Self {
        ImportAction::AddNestedImport {
            common_segments,
            path_to_split,
            first_segment_to_split,
            add_self,
        }
    }

    fn add_in_tree_list(
        common_segments: usize,
        tree_list: ast::UseTreeList,
        add_self: bool,
    ) -> Self {
        ImportAction::AddInTreeList { common_segments, tree_list, add_self }
    }

    fn better(left: ImportAction, right: ImportAction) -> ImportAction {
        if left.is_better(&right) {
            left
        } else {
            right
        }
    }

    fn is_better(&self, other: &ImportAction) -> bool {
        match (self, other) {
            (ImportAction::Nothing, _) => true,
            (ImportAction::AddInTreeList { .. }, ImportAction::Nothing) => false,
            (
                ImportAction::AddNestedImport { common_segments: n, .. },
                ImportAction::AddInTreeList { common_segments: m, .. },
            )
            | (
                ImportAction::AddInTreeList { common_segments: n, .. },
                ImportAction::AddNestedImport { common_segments: m, .. },
            )
            | (
                ImportAction::AddInTreeList { common_segments: n, .. },
                ImportAction::AddInTreeList { common_segments: m, .. },
            )
            | (
                ImportAction::AddNestedImport { common_segments: n, .. },
                ImportAction::AddNestedImport { common_segments: m, .. },
            ) => n > m,
            (ImportAction::AddInTreeList { .. }, _) => true,
            (ImportAction::AddNestedImport { .. }, ImportAction::Nothing) => false,
            (ImportAction::AddNestedImport { .. }, _) => true,
            (ImportAction::AddNewUse { .. }, _) => false,
        }
    }
}

// Find out the best ImportAction to import target path against current_use_tree.
// If current_use_tree has a nested import the function gets called recursively on every UseTree inside a UseTreeList.
fn walk_use_tree_for_best_action(
    current_path_segments: &mut Vec<ast::PathSegment>, // buffer containing path segments
    current_parent_use_tree_list: Option<ast::UseTreeList>, // will be Some value if we are in a nested import
    current_use_tree: ast::UseTree, // the use tree we are currently examinating
    target: &[SmolStr],             // the path we want to import
) -> ImportAction {
    // We save the number of segments in the buffer so we can restore the correct segments
    // before returning. Recursive call will add segments so we need to delete them.
    let prev_len = current_path_segments.len();

    let tree_list = current_use_tree.use_tree_list();
    let alias = current_use_tree.rename();

    let path = match current_use_tree.path() {
        Some(path) => path,
        None => {
            // If the use item don't have a path, it means it's broken (syntax error)
            return ImportAction::add_new_use(
                current_use_tree
                    .syntax()
                    .ancestors()
                    .find_map(ast::Use::cast)
                    .map(|it| it.syntax().clone()),
                true,
            );
        }
    };

    // This can happen only if current_use_tree is a direct child of a UseItem
    if let Some(name) = alias.and_then(|it| it.name()) {
        if compare_path_segment_with_name(&target[0], &name) {
            return ImportAction::Nothing;
        }
    }

    collect_path_segments_raw(current_path_segments, path.clone());

    // We compare only the new segments added in the line just above.
    // The first prev_len segments were already compared in 'parent' recursive calls.
    let left = target.split_at(prev_len).1;
    let right = current_path_segments.split_at(prev_len).1;
    let common = compare_path_segments(left, &right);
    let mut action = match common {
        0 => ImportAction::add_new_use(
            // e.g: target is std::fmt and we can have
            // use foo::bar
            // We add a brand new use statement
            current_use_tree
                .syntax()
                .ancestors()
                .find_map(ast::Use::cast)
                .map(|it| it.syntax().clone()),
            true,
        ),
        common if common == left.len() && left.len() == right.len() => {
            // e.g: target is std::fmt and we can have
            // 1- use std::fmt;
            // 2- use std::fmt::{ ... }
            if let Some(list) = tree_list {
                // In case 2 we need to add self to the nested list
                // unless it's already there
                let has_self = list.use_trees().map(|it| it.path()).any(|p| {
                    p.and_then(|it| it.segment())
                        .and_then(|it| it.kind())
                        .filter(|k| *k == ast::PathSegmentKind::SelfKw)
                        .is_some()
                });

                if has_self {
                    ImportAction::Nothing
                } else {
                    ImportAction::add_in_tree_list(current_path_segments.len(), list, true)
                }
            } else {
                // Case 1
                ImportAction::Nothing
            }
        }
        common if common != left.len() && left.len() == right.len() => {
            // e.g: target is std::fmt and we have
            // use std::io;
            // We need to split.
            let segments_to_split = current_path_segments.split_at(prev_len + common).1;
            ImportAction::add_nested_import(
                prev_len + common,
                path,
                Some(segments_to_split[0].clone()),
                false,
            )
        }
        common if common == right.len() && left.len() > right.len() => {
            // e.g: target is std::fmt and we can have
            // 1- use std;
            // 2- use std::{ ... };

            // fallback action
            let mut better_action = ImportAction::add_new_use(
                current_use_tree
                    .syntax()
                    .ancestors()
                    .find_map(ast::Use::cast)
                    .map(|it| it.syntax().clone()),
                true,
            );
            if let Some(list) = tree_list {
                // Case 2, check recursively if the path is already imported in the nested list
                for u in list.use_trees() {
                    let child_action = walk_use_tree_for_best_action(
                        current_path_segments,
                        Some(list.clone()),
                        u,
                        target,
                    );
                    if child_action.is_better(&better_action) {
                        better_action = child_action;
                        if let ImportAction::Nothing = better_action {
                            return better_action;
                        }
                    }
                }
            } else {
                // Case 1, split adding self
                better_action = ImportAction::add_nested_import(prev_len + common, path, None, true)
            }
            better_action
        }
        common if common == left.len() && left.len() < right.len() => {
            // e.g: target is std::fmt and we can have
            // use std::fmt::Debug;
            let segments_to_split = current_path_segments.split_at(prev_len + common).1;
            ImportAction::add_nested_import(
                prev_len + common,
                path,
                Some(segments_to_split[0].clone()),
                true,
            )
        }
        common if common < left.len() && common < right.len() => {
            // e.g: target is std::fmt::nested::Debug
            // use std::fmt::Display
            let segments_to_split = current_path_segments.split_at(prev_len + common).1;
            ImportAction::add_nested_import(
                prev_len + common,
                path,
                Some(segments_to_split[0].clone()),
                false,
            )
        }
        _ => unreachable!(),
    };

    // If we are inside a UseTreeList adding a use statement become adding to the existing
    // tree list.
    action = match (current_parent_use_tree_list, action.clone()) {
        (Some(use_tree_list), ImportAction::AddNewUse { .. }) => {
            ImportAction::add_in_tree_list(prev_len, use_tree_list, false)
        }
        (_, _) => action,
    };

    // We remove the segments added
    current_path_segments.truncate(prev_len);
    action
}

fn best_action_for_target(
    container: SyntaxNode,
    anchor: SyntaxNode,
    target: &[SmolStr],
) -> ImportAction {
    let mut storage = Vec::with_capacity(16); // this should be the only allocation
    let best_action = container
        .children()
        .filter_map(ast::Use::cast)
        .filter(|u| u.visibility().is_none())
        .filter_map(|it| it.use_tree())
        .map(|u| walk_use_tree_for_best_action(&mut storage, None, u, target))
        .fold(None, |best, a| match best {
            Some(best) => Some(ImportAction::better(best, a)),
            None => Some(a),
        });

    match best_action {
        Some(action) => action,
        None => {
            // We have no action and no UseItem was found in container so we find
            // another item and we use it as anchor.
            // If there are no items above, we choose the target path itself as anchor.
            // todo: we should include even whitespace blocks as anchor candidates
            let anchor = container.children().next().or_else(|| Some(anchor));

            let add_after_anchor = anchor
                .clone()
                .and_then(ast::Attr::cast)
                .map(|attr| attr.kind() == ast::AttrKind::Inner)
                .unwrap_or(false);
            ImportAction::add_new_use(anchor, add_after_anchor)
        }
    }
}

fn make_assist(action: &ImportAction, target: &[SmolStr], edit: &mut TextEditBuilder) {
    match action {
        ImportAction::AddNewUse { anchor, add_after_anchor } => {
            make_assist_add_new_use(anchor, *add_after_anchor, target, edit)
        }
        ImportAction::AddInTreeList { common_segments, tree_list, add_self } => {
            // We know that the fist n segments already exists in the use statement we want
            // to modify, so we want to add only the last target.len() - n segments.
            let segments_to_add = target.split_at(*common_segments).1;
            make_assist_add_in_tree_list(tree_list, segments_to_add, *add_self, edit)
        }
        ImportAction::AddNestedImport {
            common_segments,
            path_to_split,
            first_segment_to_split,
            add_self,
        } => {
            let segments_to_add = target.split_at(*common_segments).1;
            make_assist_add_nested_import(
                path_to_split,
                first_segment_to_split,
                segments_to_add,
                *add_self,
                edit,
            )
        }
        _ => {}
    }
}

fn make_assist_add_new_use(
    anchor: &Option<SyntaxNode>,
    after: bool,
    target: &[SmolStr],
    edit: &mut TextEditBuilder,
) {
    if let Some(anchor) = anchor {
        let indent = leading_indent(anchor);
        let mut buf = String::new();
        if after {
            buf.push_str("\n");
            if let Some(spaces) = &indent {
                buf.push_str(spaces);
            }
        }
        buf.push_str("use ");
        fmt_segments_raw(target, &mut buf);
        buf.push_str(";");
        if !after {
            buf.push_str("\n\n");
            if let Some(spaces) = &indent {
                buf.push_str(&spaces);
            }
        }
        let position = if after { anchor.text_range().end() } else { anchor.text_range().start() };
        edit.insert(position, buf);
    }
}

fn make_assist_add_in_tree_list(
    tree_list: &ast::UseTreeList,
    target: &[SmolStr],
    add_self: bool,
    edit: &mut TextEditBuilder,
) {
    let last = tree_list.use_trees().last();
    if let Some(last) = last {
        let mut buf = String::new();
        let comma = last.syntax().siblings(Direction::Next).find(|n| n.kind() == T![,]);
        let offset = if let Some(comma) = comma {
            comma.text_range().end()
        } else {
            buf.push_str(",");
            last.syntax().text_range().end()
        };
        if add_self {
            buf.push_str(" self")
        } else {
            buf.push_str(" ");
        }
        fmt_segments_raw(target, &mut buf);
        edit.insert(offset, buf);
    } else {
    }
}

fn make_assist_add_nested_import(
    path: &ast::Path,
    first_segment_to_split: &Option<ast::PathSegment>,
    target: &[SmolStr],
    add_self: bool,
    edit: &mut TextEditBuilder,
) {
    let use_tree = path.syntax().ancestors().find_map(ast::UseTree::cast);
    if let Some(use_tree) = use_tree {
        let (start, add_colon_colon) = if let Some(first_segment_to_split) = first_segment_to_split
        {
            (first_segment_to_split.syntax().text_range().start(), false)
        } else {
            (use_tree.syntax().text_range().end(), true)
        };
        let end = use_tree.syntax().text_range().end();

        let mut buf = String::new();
        if add_colon_colon {
            buf.push_str("::");
        }
        buf.push_str("{");
        if add_self {
            buf.push_str("self, ");
        }
        fmt_segments_raw(target, &mut buf);
        if !target.is_empty() {
            buf.push_str(", ");
        }
        edit.insert(start, buf);
        edit.insert(end, "}".to_string());
    }
}

/// If the node is on the beginning of the line, calculate indent.
fn leading_indent(node: &SyntaxNode) -> Option<SmolStr> {
    for token in prev_tokens(node.first_token()?) {
        if let Some(ws) = ast::Whitespace::cast(token.clone()) {
            let ws_text = ws.text();
            if let Some(pos) = ws_text.rfind('\n') {
                return Some(ws_text[pos + 1..].into());
            }
        }
        if token.text().contains('\n') {
            break;
        }
    }
    return None;
    fn prev_tokens(token: SyntaxToken) -> impl Iterator<Item = SyntaxToken> {
        successors(token.prev_token(), |token| token.prev_token())
    }
}
