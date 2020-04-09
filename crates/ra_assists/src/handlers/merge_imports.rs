use std::iter::successors;

use ra_syntax::{
    algo::{neighbor, SyntaxRewriter},
    ast::{self, edit::AstNodeEdit, make},
    AstNode, AstToken, Direction, InsertPosition, SyntaxElement, T,
};

use crate::{Assist, AssistCtx, AssistId};

// Assist: merge_imports
//
// Merges two imports with a common prefix.
//
// ```
// use std::<|>fmt::Formatter;
// use std::io;
// ```
// ->
// ```
// use std::{fmt::Formatter, io};
// ```
pub(crate) fn merge_imports(ctx: AssistCtx) -> Option<Assist> {
    let tree: ast::UseTree = ctx.find_node_at_offset()?;
    let mut rewriter = SyntaxRewriter::default();
    let mut offset = ctx.frange.range.start();

    if let Some(use_item) = tree.syntax().parent().and_then(ast::UseItem::cast) {
        let (merged, to_delete) = next_prev()
            .filter_map(|dir| neighbor(&use_item, dir))
            .filter_map(|it| Some((it.clone(), it.use_tree()?)))
            .find_map(|(use_item, use_tree)| {
                Some((try_merge_trees(&tree, &use_tree)?, use_item.clone()))
            })?;

        rewriter.replace_ast(&tree, &merged);
        rewriter += to_delete.remove();

        if to_delete.syntax().text_range().end() < offset {
            offset -= to_delete.syntax().text_range().len();
        }
    } else {
        let (merged, to_delete) = next_prev()
            .filter_map(|dir| neighbor(&tree, dir))
            .find_map(|use_tree| Some((try_merge_trees(&tree, &use_tree)?, use_tree.clone())))?;

        rewriter.replace_ast(&tree, &merged);
        rewriter += to_delete.remove();

        if to_delete.syntax().text_range().end() < offset {
            offset -= to_delete.syntax().text_range().len();
        }
    };

    ctx.add_assist(AssistId("merge_imports"), "Merge imports", |edit| {
        edit.rewrite(rewriter);
        // FIXME: we only need because our diff is imprecise
        edit.set_cursor(offset);
    })
}

fn next_prev() -> impl Iterator<Item = Direction> {
    [Direction::Next, Direction::Prev].iter().copied()
}

fn try_merge_trees(old: &ast::UseTree, new: &ast::UseTree) -> Option<ast::UseTree> {
    let lhs_path = old.path()?;
    let rhs_path = new.path()?;

    let (lhs_prefix, rhs_prefix) = common_prefix(&lhs_path, &rhs_path)?;

    let lhs = old.split_prefix(&lhs_prefix);
    let rhs = new.split_prefix(&rhs_prefix);

    let mut to_insert: Vec<SyntaxElement> = Vec::new();
    to_insert.push(make::token(T![,]).into());
    to_insert.push(make::tokens::single_space().into());
    to_insert.extend(
        rhs.use_tree_list()?
            .syntax()
            .children_with_tokens()
            .filter(|it| it.kind() != T!['{'] && it.kind() != T!['}']),
    );
    let use_tree_list = lhs.use_tree_list()?;
    let pos = InsertPosition::Before(use_tree_list.r_curly()?.syntax().clone().into());
    let use_tree_list = use_tree_list.insert_children(pos, to_insert);
    Some(lhs.with_use_tree_list(use_tree_list))
}

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

        match (lhs_curr.parent_path(), rhs_curr.parent_path()) {
            (Some(lhs), Some(rhs)) => {
                lhs_curr = lhs;
                rhs_curr = rhs;
            }
            _ => break,
        }
    }

    res
}

fn first_path(path: &ast::Path) -> ast::Path {
    successors(Some(path.clone()), |it| it.qualifier()).last().unwrap()
}

#[cfg(test)]
mod tests {
    use crate::helpers::check_assist;

    use super::*;

    #[test]
    fn test_merge_first() {
        check_assist(
            merge_imports,
            r"
use std::fmt<|>::Debug;
use std::fmt::Display;
",
            r"
use std::fmt<|>::{Debug, Display};
",
        )
    }

    #[test]
    fn test_merge_second() {
        check_assist(
            merge_imports,
            r"
use std::fmt::Debug;
use std::fmt<|>::Display;
",
            r"
use std::fmt:<|>:{Display, Debug};
",
        );
    }

    #[test]
    fn test_merge_nested() {
        check_assist(
            merge_imports,
            r"
use std::{fmt<|>::Debug, fmt::Display};
",
            r"
use std::{fmt<|>::{Debug, Display}};
",
        );
        check_assist(
            merge_imports,
            r"
use std::{fmt::Debug, fmt<|>::Display};
",
            r"
use std::{fmt::<|>{Display, Debug}};
",
        );
    }

    #[test]
    fn test_merge_single_wildcard_diff_prefixes() {
        check_assist(
            merge_imports,
            r"
use std<|>::cell::*;
use std::str;
",
            r"
use std<|>::{cell::*, str};
",
        )
    }

    #[test]
    fn test_merge_both_wildcard_diff_prefixes() {
        check_assist(
            merge_imports,
            r"
use std<|>::cell::*;
use std::str::*;
",
            r"
use std<|>::{cell::*, str::*};
",
        )
    }

    #[test]
    fn removes_just_enough_whitespace() {
        check_assist(
            merge_imports,
            r"
use foo<|>::bar;
use foo::baz;

/// Doc comment
",
            r"
use foo<|>::{bar, baz};

/// Doc comment
",
        );
    }

    #[test]
    fn works_with_trailing_comma() {
        check_assist(
            merge_imports,
            r"
use {
    foo<|>::bar,
    foo::baz,
};
",
            r"
use {
    foo<|>::{bar, baz},
};
",
        );
        check_assist(
            merge_imports,
            r"
use {
    foo::baz,
    foo<|>::bar,
};
",
            r"
use {
    foo::{bar<|>, baz},
};
",
        );
    }
}
