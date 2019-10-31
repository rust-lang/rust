use std::iter::successors;

use hir::db::HirDatabase;
use ra_syntax::{ast, AstNode, TextUnit, T};

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
pub(crate) fn split_import(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let colon_colon = ctx.find_token_at_offset(T![::])?;
    let path = ast::Path::cast(colon_colon.parent())?;
    let top_path = successors(Some(path), |it| it.parent_path()).last()?;

    let use_tree = top_path.syntax().ancestors().find_map(ast::UseTree::cast);
    if use_tree.is_none() {
        return None;
    }

    let l_curly = colon_colon.text_range().end();
    let r_curly = match top_path.syntax().parent().and_then(ast::UseTree::cast) {
        Some(tree) => tree.syntax().text_range().end(),
        None => top_path.syntax().text_range().end(),
    };

    ctx.add_assist(AssistId("split_import"), "split import", |edit| {
        edit.target(colon_colon.text_range());
        edit.insert(l_curly, "{");
        edit.insert(r_curly, "}");
        edit.set_cursor(l_curly + TextUnit::of_str("{"));
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{check_assist, check_assist_target};

    #[test]
    fn test_split_import() {
        check_assist(
            split_import,
            "use crate::<|>db::RootDatabase;",
            "use crate::{<|>db::RootDatabase};",
        )
    }

    #[test]
    fn split_import_works_with_trees() {
        check_assist(
            split_import,
            "use crate:<|>:db::{RootDatabase, FileSymbol}",
            "use crate::{<|>db::{RootDatabase, FileSymbol}}",
        )
    }

    #[test]
    fn split_import_target() {
        check_assist_target(split_import, "use crate::<|>db::{RootDatabase, FileSymbol}", "::");
    }
}
