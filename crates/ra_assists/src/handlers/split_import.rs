use std::iter::{once, successors};

use ra_syntax::{
    ast::{self, make},
    AstNode, T,
};

use crate::{Assist, AssistCtx, AssistId};

// Assist: split_import
//
// Wraps the tail of import into braces.
//
// ```
// use std::<|>collections::HashMap;
// ```
// ->
// ```
// use std::{collections::HashMap};
// ```
pub(crate) fn split_import(ctx: AssistCtx) -> Option<Assist> {
    let colon_colon = ctx.find_token_at_offset(T![::])?;
    let path = ast::Path::cast(colon_colon.parent())?.qualifier()?;
    let top_path = successors(Some(path.clone()), |it| it.parent_path()).last()?;

    let use_tree = top_path.syntax().ancestors().find_map(ast::UseTree::cast)?;

    let new_tree = split_use_tree_prefix(&use_tree, &path)?;
    let cursor = ctx.frange.range.start();

    ctx.add_assist(AssistId("split_import"), "Split import", |edit| {
        edit.target(colon_colon.text_range());
        edit.replace_ast(use_tree, new_tree);
        edit.set_cursor(cursor);
    })
}

fn split_use_tree_prefix(use_tree: &ast::UseTree, prefix: &ast::Path) -> Option<ast::UseTree> {
    let suffix = split_path_prefix(&prefix)?;
    let use_tree = make::use_tree(suffix.clone(), use_tree.use_tree_list(), use_tree.alias());
    let nested = make::use_tree_list(once(use_tree));
    let res = make::use_tree(prefix.clone(), Some(nested), None);
    Some(res)
}

fn split_path_prefix(prefix: &ast::Path) -> Option<ast::Path> {
    let parent = prefix.parent_path()?;
    let mut res = make::path_unqualified(parent.segment()?);
    for p in successors(parent.parent_path(), |it| it.parent_path()) {
        res = make::path_qualified(res, p.segment()?);
    }
    Some(res)
}

#[cfg(test)]
mod tests {
    use crate::helpers::{check_assist, check_assist_target};

    use super::*;

    #[test]
    fn test_split_import() {
        check_assist(
            split_import,
            "use crate::<|>db::RootDatabase;",
            "use crate::<|>{db::RootDatabase};",
        )
    }

    #[test]
    fn split_import_works_with_trees() {
        check_assist(
            split_import,
            "use crate:<|>:db::{RootDatabase, FileSymbol}",
            "use crate:<|>:{db::{RootDatabase, FileSymbol}}",
        )
    }

    #[test]
    fn split_import_target() {
        check_assist_target(split_import, "use crate::<|>db::{RootDatabase, FileSymbol}", "::");
    }
}
