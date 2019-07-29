use hir::{self, db::HirDatabase};
use ra_text_edit::TextEditBuilder;

use crate::{
    assist_ctx::{Assist, AssistCtx},
    AssistId,
};
use ra_syntax::{
    ast::{self, NameOwner},
    AstNode, Direction, SmolStr,
    SyntaxKind::{PATH, PATH_SEGMENT},
    SyntaxNode, TextRange, T,
};

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

fn fmt_segments(segments: &[SmolStr]) -> String {
    let mut buf = String::new();
    fmt_segments_raw(segments, &mut buf);
    buf
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

// Returns the numeber of common segments.
fn compare_path_segments(left: &[SmolStr], right: &[ast::PathSegment]) -> usize {
    left.iter().zip(right).filter(|(l, r)| compare_path_segment(l, r)).count()
}

fn compare_path_segment(a: &SmolStr, b: &ast::PathSegment) -> bool {
    if let Some(kb) = b.kind() {
        match kb {
            ast::PathSegmentKind::Name(nameref_b) => a == nameref_b.text(),
            ast::PathSegmentKind::SelfKw => a == "self",
            ast::PathSegmentKind::SuperKw => a == "super",
            ast::PathSegmentKind::CrateKw => a == "crate",
        }
    } else {
        false
    }
}

fn compare_path_segment_with_name(a: &SmolStr, b: &ast::Name) -> bool {
    a == b.text()
}

#[derive(Clone)]
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
            ) => n > m,
            (
                ImportAction::AddInTreeList { common_segments: n, .. },
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
    let alias = current_use_tree.alias();

    let path = match current_use_tree.path() {
        Some(path) => path,
        None => {
            // If the use item don't have a path, it means it's broken (syntax error)
            return ImportAction::add_new_use(
                current_use_tree
                    .syntax()
                    .ancestors()
                    .find_map(ast::UseItem::cast)
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
                .find_map(ast::UseItem::cast)
                .map(|it| it.syntax().clone()),
            true,
        ),
        common if common == left.len() && left.len() == right.len() => {
            // e.g: target is std::fmt and we can have
            // 1- use std::fmt;
            // 2- use std::fmt:{ ... }
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
                    .find_map(ast::UseItem::cast)
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
        .filter_map(ast::UseItem::cast)
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
            let anchor = container
                .children()
                .find(|n| n.text_range().start() < anchor.text_range().start())
                .or_else(|| Some(anchor));

            ImportAction::add_new_use(anchor, false)
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
        let indent = ra_fmt::leading_indent(anchor);
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
        buf.push_str("{ ");
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

fn apply_auto_import(
    container: &SyntaxNode,
    path: &ast::Path,
    target: &[SmolStr],
    edit: &mut TextEditBuilder,
) {
    let action = best_action_for_target(container.clone(), path.syntax().clone(), target);
    make_assist(&action, target, edit);
    if let Some(last) = path.segment() {
        // Here we are assuming the assist will provide a  correct use statement
        // so we can delete the path qualifier
        edit.delete(TextRange::from_to(
            path.syntax().text_range().start(),
            last.syntax().text_range().start(),
        ));
    }
}

pub fn collect_hir_path_segments(path: &hir::Path) -> Vec<SmolStr> {
    let mut ps = Vec::<SmolStr>::with_capacity(10);
    match path.kind {
        hir::PathKind::Abs => ps.push("".into()),
        hir::PathKind::Crate => ps.push("crate".into()),
        hir::PathKind::Plain => {}
        hir::PathKind::Self_ => ps.push("self".into()),
        hir::PathKind::Super => ps.push("super".into()),
    }
    for s in path.segments.iter() {
        ps.push(s.name.to_string().into());
    }
    ps
}

// This function produces sequence of text edits into edit
// to import the target path in the most appropriate scope given
// the cursor position
pub fn auto_import_text_edit(
    // Ideally the position of the cursor, used to
    position: &SyntaxNode,
    // The statement to use as anchor (last resort)
    anchor: &SyntaxNode,
    // The path to import as a sequence of strings
    target: &[SmolStr],
    edit: &mut TextEditBuilder,
) {
    let container = position.ancestors().find_map(|n| {
        if let Some(module) = ast::Module::cast(n.clone()) {
            return module.item_list().map(|it| it.syntax().clone());
        }
        ast::SourceFile::cast(n).map(|it| it.syntax().clone())
    });

    if let Some(container) = container {
        let action = best_action_for_target(container, anchor.clone(), target);
        make_assist(&action, target, edit);
    }
}

pub(crate) fn auto_import(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let path: ast::Path = ctx.node_at_offset()?;
    // We don't want to mess with use statements
    if path.syntax().ancestors().find_map(ast::UseItem::cast).is_some() {
        return None;
    }

    let hir_path = hir::Path::from_ast(path.clone())?;
    let segments = collect_hir_path_segments(&hir_path);
    if segments.len() < 2 {
        return None;
    }

    if let Some(module) = path.syntax().ancestors().find_map(ast::Module::cast) {
        if let (Some(item_list), Some(name)) = (module.item_list(), module.name()) {
            ctx.add_action(
                AssistId("auto_import"),
                format!("import {} in mod {}", fmt_segments(&segments), name.text()),
                |edit| {
                    apply_auto_import(
                        item_list.syntax(),
                        &path,
                        &segments,
                        edit.text_edit_builder(),
                    );
                },
            );
        }
    } else {
        let current_file = path.syntax().ancestors().find_map(ast::SourceFile::cast)?;
        ctx.add_action(
            AssistId("auto_import"),
            format!("import {} in the current file", fmt_segments(&segments)),
            |edit| {
                apply_auto_import(
                    current_file.syntax(),
                    &path,
                    &segments,
                    edit.text_edit_builder(),
                );
            },
        );
    }

    ctx.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_auto_import_add_use_no_anchor() {
        check_assist(
            auto_import,
            "
std::fmt::Debug<|>
    ",
            "
use std::fmt::Debug;

Debug<|>
    ",
        );
    }
    #[test]
    fn test_auto_import_add_use_no_anchor_with_item_below() {
        check_assist(
            auto_import,
            "
std::fmt::Debug<|>

fn main() {
}
    ",
            "
use std::fmt::Debug;

Debug<|>

fn main() {
}
    ",
        );
    }

    #[test]
    fn test_auto_import_add_use_no_anchor_with_item_above() {
        check_assist(
            auto_import,
            "
fn main() {
}

std::fmt::Debug<|>
    ",
            "
use std::fmt::Debug;

fn main() {
}

Debug<|>
    ",
        );
    }

    #[test]
    fn test_auto_import_add_use_no_anchor_2seg() {
        check_assist(
            auto_import,
            "
std::fmt<|>::Debug
    ",
            "
use std::fmt;

fmt<|>::Debug
    ",
        );
    }

    #[test]
    fn test_auto_import_add_use() {
        check_assist(
            auto_import,
            "
use stdx;

impl std::fmt::Debug<|> for Foo {
}
    ",
            "
use stdx;
use std::fmt::Debug;

impl Debug<|> for Foo {
}
    ",
        );
    }

    #[test]
    fn test_auto_import_file_use_other_anchor() {
        check_assist(
            auto_import,
            "
impl std::fmt::Debug<|> for Foo {
}
    ",
            "
use std::fmt::Debug;

impl Debug<|> for Foo {
}
    ",
        );
    }

    #[test]
    fn test_auto_import_add_use_other_anchor_indent() {
        check_assist(
            auto_import,
            "
    impl std::fmt::Debug<|> for Foo {
    }
    ",
            "
    use std::fmt::Debug;

    impl Debug<|> for Foo {
    }
    ",
        );
    }

    #[test]
    fn test_auto_import_split_different() {
        check_assist(
            auto_import,
            "
use std::fmt;

impl std::io<|> for Foo {
}
    ",
            "
use std::{ io, fmt};

impl io<|> for Foo {
}
    ",
        );
    }

    #[test]
    fn test_auto_import_split_self_for_use() {
        check_assist(
            auto_import,
            "
use std::fmt;

impl std::fmt::Debug<|> for Foo {
}
    ",
            "
use std::fmt::{ self, Debug, };

impl Debug<|> for Foo {
}
    ",
        );
    }

    #[test]
    fn test_auto_import_split_self_for_target() {
        check_assist(
            auto_import,
            "
use std::fmt::Debug;

impl std::fmt<|> for Foo {
}
    ",
            "
use std::fmt::{ self, Debug};

impl fmt<|> for Foo {
}
    ",
        );
    }

    #[test]
    fn test_auto_import_add_to_nested_self_nested() {
        check_assist(
            auto_import,
            "
use std::fmt::{Debug, nested::{Display}};

impl std::fmt::nested<|> for Foo {
}
",
            "
use std::fmt::{Debug, nested::{Display, self}};

impl nested<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_auto_import_add_to_nested_self_already_included() {
        check_assist(
            auto_import,
            "
use std::fmt::{Debug, nested::{self, Display}};

impl std::fmt::nested<|> for Foo {
}
",
            "
use std::fmt::{Debug, nested::{self, Display}};

impl nested<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_auto_import_add_to_nested_nested() {
        check_assist(
            auto_import,
            "
use std::fmt::{Debug, nested::{Display}};

impl std::fmt::nested::Debug<|> for Foo {
}
",
            "
use std::fmt::{Debug, nested::{Display, Debug}};

impl Debug<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_auto_import_split_common_target_longer() {
        check_assist(
            auto_import,
            "
use std::fmt::Debug;

impl std::fmt::nested::Display<|> for Foo {
}
",
            "
use std::fmt::{ nested::Display, Debug};

impl Display<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_auto_import_split_common_use_longer() {
        check_assist(
            auto_import,
            "
use std::fmt::nested::Debug;

impl std::fmt::Display<|> for Foo {
}
",
            "
use std::fmt::{ Display, nested::Debug};

impl Display<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_auto_import_alias() {
        check_assist(
            auto_import,
            "
use std::fmt as foo;

impl foo::Debug<|> for Foo {
}
",
            "
use std::fmt as foo;

impl Debug<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_auto_import_not_applicable_one_segment() {
        check_assist_not_applicable(
            auto_import,
            "
impl foo<|> for Foo {
}
",
        );
    }

    #[test]
    fn test_auto_import_not_applicable_in_use() {
        check_assist_not_applicable(
            auto_import,
            "
use std::fmt<|>;
",
        );
    }

    #[test]
    fn test_auto_import_add_use_no_anchor_in_mod_mod() {
        check_assist(
            auto_import,
            "
mod foo {
    mod bar {
        std::fmt::Debug<|>
    }
}
    ",
            "
mod foo {
    mod bar {
        use std::fmt::Debug;

        Debug<|>
    }
}
    ",
        );
    }
}
