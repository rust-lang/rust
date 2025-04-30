use syntax::{AstNode, T, ast};

use crate::{AssistContext, AssistId, Assists};

// Assist: split_import
//
// Wraps the tail of import into braces.
//
// ```
// use std::$0collections::HashMap;
// ```
// ->
// ```
// use std::{collections::HashMap};
// ```
pub(crate) fn split_import(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let colon_colon = ctx.find_token_syntax_at_offset(T![::])?;
    let path = ast::Path::cast(colon_colon.parent()?)?.qualifier()?;

    let use_tree = path.top_path().syntax().ancestors().find_map(ast::UseTree::cast)?;

    let has_errors = use_tree
        .syntax()
        .descendants_with_tokens()
        .any(|it| it.kind() == syntax::SyntaxKind::ERROR);
    let last_segment = use_tree.path().and_then(|it| it.segment());
    if has_errors || last_segment.is_none() {
        return None;
    }

    let target = colon_colon.text_range();
    acc.add(AssistId::refactor_rewrite("split_import"), "Split import", target, |edit| {
        let use_tree = edit.make_mut(use_tree.clone());
        let path = edit.make_mut(path);
        use_tree.split_prefix(&path);
    })
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    fn test_split_import() {
        check_assist(
            split_import,
            "use crate::$0db::RootDatabase;",
            "use crate::{db::RootDatabase};",
        )
    }

    #[test]
    fn split_import_works_with_trees() {
        check_assist(
            split_import,
            "use crate:$0:db::{RootDatabase, FileSymbol}",
            "use crate::{db::{RootDatabase, FileSymbol}}",
        )
    }

    #[test]
    fn split_import_target() {
        check_assist_target(split_import, "use crate::$0db::{RootDatabase, FileSymbol}", "::");
    }

    #[test]
    fn issue4044() {
        check_assist_not_applicable(split_import, "use crate::$0:::self;")
    }

    #[test]
    fn test_empty_use() {
        check_assist_not_applicable(
            split_import,
            r"
use std::$0
fn main() {}",
        );
    }
}
