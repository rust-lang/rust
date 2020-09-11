//! Handle syntactic aspects of inserting a new `use`.
use std::iter::{self, successors};

use algo::skip_trivia_token;
use ast::{
    edit::{AstNodeEdit, IndentLevel},
    PathSegmentKind, VisibilityOwner,
};
use syntax::{
    algo,
    ast::{self, make, AstNode},
    Direction, InsertPosition, SyntaxElement, SyntaxNode, T,
};
use test_utils::mark;

#[derive(Debug)]
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
    pub(crate) fn find_insert_use_container(
        position: &SyntaxNode,
        ctx: &crate::assist_context::AssistContext,
    ) -> Option<Self> {
        ctx.sema.ancestors_with_macros(position.clone()).find_map(Self::from)
    }

    pub(crate) fn as_syntax_node(&self) -> &SyntaxNode {
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

    fn insert_pos_after_inner_attribute(&self) -> (InsertPosition<SyntaxElement>, AddBlankLine) {
        // check if the scope has inner attributes, we dont want to insert in front of them
        match self
            .as_syntax_node()
            .children()
            // no flat_map here cause we want to short circuit the iterator
            .map(ast::Attr::cast)
            .take_while(|attr| {
                attr.as_ref().map(|attr| attr.kind() == ast::AttrKind::Inner).unwrap_or(false)
            })
            .last()
            .flatten()
        {
            Some(attr) => {
                (InsertPosition::After(attr.syntax().clone().into()), AddBlankLine::BeforeTwice)
            }
            None => self.first_insert_pos(),
        }
    }
}

/// Insert an import path into the given file/node. A `merge` value of none indicates that no import merging is allowed to occur.
pub(crate) fn insert_use(
    scope: &ImportScope,
    path: ast::Path,
    merge: Option<MergeBehaviour>,
) -> SyntaxNode {
    let use_item = make::use_(make::use_tree(path.clone(), None, None, false));
    // merge into existing imports if possible
    if let Some(mb) = merge {
        for existing_use in scope.as_syntax_node().children().filter_map(ast::Use::cast) {
            if let Some(merged) = try_merge_imports(&existing_use, &use_item, mb) {
                let to_delete: SyntaxElement = existing_use.syntax().clone().into();
                let to_delete = to_delete.clone()..=to_delete;
                let to_insert = iter::once(merged.syntax().clone().into());
                return algo::replace_children(scope.as_syntax_node(), to_delete, to_insert);
            }
        }
    }

    // either we weren't allowed to merge or there is no import that fits the merge conditions
    // so look for the place we have to insert to
    let (insert_position, add_blank) = find_insert_position(scope, path);

    let to_insert: Vec<SyntaxElement> = {
        let mut buf = Vec::new();

        match add_blank {
            AddBlankLine::Before | AddBlankLine::Around => {
                buf.push(make::tokens::single_newline().into())
            }
            AddBlankLine::BeforeTwice => buf.push(make::tokens::blank_line().into()),
            _ => (),
        }

        if let ident_level @ 1..=usize::MAX = scope.indent_level().0 as usize {
            // FIXME: this alone doesnt properly re-align all cases
            buf.push(make::tokens::whitespace(&" ".repeat(4 * ident_level)).into());
        }
        buf.push(use_item.syntax().clone().into());

        match add_blank {
            AddBlankLine::After | AddBlankLine::Around => {
                buf.push(make::tokens::single_newline().into())
            }
            AddBlankLine::AfterTwice => buf.push(make::tokens::blank_line().into()),
            _ => (),
        }

        buf
    };

    algo::insert_children(scope.as_syntax_node(), insert_position, to_insert)
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

pub(crate) fn try_merge_imports(
    old: &ast::Use,
    new: &ast::Use,
    merge_behaviour: MergeBehaviour,
) -> Option<ast::Use> {
    // don't merge imports with different visibilities
    if !eq_visibility(old.visibility(), new.visibility()) {
        return None;
    }
    let old_tree = old.use_tree()?;
    let new_tree = new.use_tree()?;
    let merged = try_merge_trees(&old_tree, &new_tree, merge_behaviour)?;
    Some(old.with_use_tree(merged))
}

/// Simple function that checks if a UseTreeList is deeper than one level
fn use_tree_list_is_nested(tl: &ast::UseTreeList) -> bool {
    tl.use_trees().any(|use_tree| {
        use_tree.use_tree_list().is_some() || use_tree.path().and_then(|p| p.qualifier()).is_some()
    })
}

// FIXME: currently this merely prepends the new tree into old, ideally it would insert the items in a sorted fashion
pub(crate) fn try_merge_trees(
    old: &ast::UseTree,
    new: &ast::UseTree,
    merge_behaviour: MergeBehaviour,
) -> Option<ast::UseTree> {
    let lhs_path = old.path()?;
    let rhs_path = new.path()?;

    let (lhs_prefix, rhs_prefix) = common_prefix(&lhs_path, &rhs_path)?;
    let lhs = old.split_prefix(&lhs_prefix);
    let rhs = new.split_prefix(&rhs_prefix);
    let lhs_tl = lhs.use_tree_list()?;
    let rhs_tl = rhs.use_tree_list()?;

    // if we are only allowed to merge the last level check if the split off paths are only one level deep
    if merge_behaviour == MergeBehaviour::Last
        && (use_tree_list_is_nested(&lhs_tl) || use_tree_list_is_nested(&rhs_tl))
    {
        mark::hit!(test_last_merge_too_long);
        return None;
    }

    let should_insert_comma = lhs_tl
        .r_curly_token()
        .and_then(|it| skip_trivia_token(it.prev_token()?, Direction::Prev))
        .map(|it| it.kind())
        != Some(T![,]);
    let mut to_insert: Vec<SyntaxElement> = Vec::new();
    if should_insert_comma {
        to_insert.push(make::token(T![,]).into());
        to_insert.push(make::tokens::single_space().into());
    }
    to_insert.extend(
        rhs_tl.syntax().children_with_tokens().filter(|it| !matches!(it.kind(), T!['{'] | T!['}'])),
    );
    let pos = InsertPosition::Before(lhs_tl.r_curly_token()?.into());
    let use_tree_list = lhs_tl.insert_children(pos, to_insert);
    Some(lhs.with_use_tree_list(use_tree_list))
}

/// Traverses both paths until they differ, returning the common prefix of both.
fn common_prefix(lhs: &ast::Path, rhs: &ast::Path) -> Option<(ast::Path, ast::Path)> {
    let mut res = None;
    let mut lhs_curr = first_path(&lhs);
    let mut rhs_curr = first_path(&rhs);
    loop {
        match (lhs_curr.segment(), rhs_curr.segment()) {
            (Some(lhs), Some(rhs)) if lhs.syntax().text() == rhs.syntax().text() => (),
            _ => break,
        }
        res = Some((lhs_curr.clone(), rhs_curr.clone()));

        match lhs_curr.parent_path().zip(rhs_curr.parent_path()) {
            Some((lhs, rhs)) => {
                lhs_curr = lhs;
                rhs_curr = rhs;
            }
            _ => break,
        }
    }

    res
}

/// What type of merges are allowed.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum MergeBehaviour {
    /// Merge everything together creating deeply nested imports.
    Full,
    /// Only merge the last import level, doesn't allow import nesting.
    Last,
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
                // FIXME: can be ThisModule as well
                _ => ImportGroup::ExternCrate,
            },
            PathSegmentKind::Type { .. } => unreachable!(),
        }
    }
}

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

#[derive(PartialEq, Eq)]
enum AddBlankLine {
    Before,
    BeforeTwice,
    Around,
    After,
    AfterTwice,
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
        .flat_map(|(use_, node)| use_.use_tree().and_then(|tree| tree.path()).zip(Some(node)));
    // Iterator that discards anything thats not in the required grouping
    // This implementation allows the user to rearrange their import groups as this only takes the first group that fits
    let group_iter = path_node_iter
        .clone()
        .skip_while(|(path, _)| ImportGroup::new(path) != group)
        .take_while(|(path, _)| ImportGroup::new(path) == group);

    let segments = segment_iter(&insert_path);
    // track the last element we iterated over, if this is still None after the iteration then that means we never iterated in the first place
    let mut last = None;
    // find the element that would come directly after our new import
    let post_insert =
        group_iter.inspect(|(_, node)| last = Some(node.clone())).find(|(path, _)| {
            let check_segments = segment_iter(&path);
            segments
                .clone()
                .zip(check_segments)
                .flat_map(|(seg, seg2)| seg.name_ref().zip(seg2.name_ref()))
                .all(|(l, r)| l.text() <= r.text())
        });
    match post_insert {
        // insert our import before that element
        Some((_, node)) => (InsertPosition::Before(node.into()), AddBlankLine::After),
        // there is no element after our new import, so append it to the end of the group
        None => match last {
            Some(node) => (InsertPosition::After(node.into()), AddBlankLine::Before),
            // the group we were looking for actually doesnt exist, so insert
            None => {
                // similar concept here to the `last` from above
                let mut last = None;
                // find the group that comes after where we want to insert
                let post_group = path_node_iter
                    .inspect(|(_, node)| last = Some(node.clone()))
                    .find(|(p, _)| ImportGroup::new(p) > group);
                match post_group {
                    Some((_, node)) => {
                        (InsertPosition::Before(node.into()), AddBlankLine::AfterTwice)
                    }
                    // there is no such group, so append after the last one
                    None => match last {
                        Some(node) => {
                            (InsertPosition::After(node.into()), AddBlankLine::BeforeTwice)
                        }
                        // there are no imports in this file at all
                        None => scope.insert_pos_after_inner_attribute(),
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

fn main() {}",
            r"#![allow(unused_imports)]

use foo::bar;

fn main() {}",
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
            r"use std::foo::bar::{Qux, Baz};",
        )
    }

    #[test]
    fn merge_groups_long_last() {
        check_last(
            "std::foo::bar::Baz",
            r"use std::foo::bar::Qux;",
            r"use std::foo::bar::{Qux, Baz};",
        )
    }

    #[test]
    fn merge_groups_long_full_list() {
        check_full(
            "std::foo::bar::Baz",
            r"use std::foo::bar::{Qux, Quux};",
            r"use std::foo::bar::{Qux, Quux, Baz};",
        )
    }

    #[test]
    fn merge_groups_long_last_list() {
        check_last(
            "std::foo::bar::Baz",
            r"use std::foo::bar::{Qux, Quux};",
            r"use std::foo::bar::{Qux, Quux, Baz};",
        )
    }

    #[test]
    fn merge_groups_long_full_nested() {
        check_full(
            "std::foo::bar::Baz",
            r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
            r"use std::foo::bar::{Qux, quux::{Fez, Fizz}, Baz};",
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
            r"use std::{self, fmt::Result};
use std::io;",
        )
    }

    #[test]
    fn merge_groups_self() {
        check_full("std::fmt::Debug", r"use std::fmt;", r"use std::fmt::{self, Debug};")
    }

    #[test]
    fn merge_self_glob() {
        check_full(
            "token::TokenKind",
            r"use token::TokenKind::*;",
            r"use token::TokenKind::{self::*, self};",
        )
    }

    #[test]
    fn merge_last_too_long() {
        mark::check!(test_last_merge_too_long);
        check_last(
            "foo::bar",
            r"use foo::bar::baz::Qux;",
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

    fn check(
        path: &str,
        ra_fixture_before: &str,
        ra_fixture_after: &str,
        mb: Option<MergeBehaviour>,
    ) {
        let file = super::ImportScope::from(
            ast::SourceFile::parse(ra_fixture_before).tree().syntax().clone(),
        )
        .unwrap();
        let path = ast::SourceFile::parse(&format!("use {};", path))
            .tree()
            .syntax()
            .descendants()
            .find_map(ast::Path::cast)
            .unwrap();

        let result = insert_use(&file, path, mb).to_string();
        assert_eq_text!(&result, ra_fixture_after);
    }

    fn check_full(path: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
        check(path, ra_fixture_before, ra_fixture_after, Some(MergeBehaviour::Full))
    }

    fn check_last(path: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
        check(path, ra_fixture_before, ra_fixture_after, Some(MergeBehaviour::Last))
    }

    fn check_none(path: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
        check(path, ra_fixture_before, ra_fixture_after, None)
    }
}
