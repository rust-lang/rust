use std::iter::successors;

use ra_syntax::{ast, AstNode, T};

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

    let new_tree = use_tree.split_prefix(&path);
    if new_tree == use_tree {
        return None;
    }
    let cursor = ctx.frange.range.start();

    ctx.add_assist(AssistId("split_import"), "Split import", |edit| {
        edit.target(colon_colon.text_range());
        edit.replace_ast(use_tree, new_tree);
        edit.set_cursor(cursor);
    })
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
