use syntax::{
    ast::{self, VisibilityOwner},
    ted::{self, Position},
    AstNode, SyntaxKind,
};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: unmerge_use
//
// Extracts single use item from use list.
//
// ```
// use std::fmt::{Debug, Display$0};
// ```
// ->
// ```
// use std::fmt::{Debug};
// use std::fmt::Display;
// ```
pub(crate) fn unmerge_use(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let tree: ast::UseTree = ctx.find_node_at_offset::<ast::UseTree>()?.clone_for_update();

    let tree_list = tree.syntax().parent().and_then(ast::UseTreeList::cast)?;
    if tree_list.use_trees().count() < 2 {
        cov_mark::hit!(skip_single_use_item);
        return None;
    }

    let use_: ast::Use = tree_list.syntax().ancestors().find_map(ast::Use::cast)?;
    let path = resolve_full_path(&tree)?;

    let old_parent_range = use_.syntax().parent()?.text_range();
    let new_parent = use_.syntax().parent()?;

    let target = tree.syntax().text_range();
    acc.add(
        AssistId("unmerge_use", AssistKind::RefactorRewrite),
        "Unmerge use",
        target,
        |builder| {
            let new_use = ast::make::use_(
                use_.visibility(),
                ast::make::use_tree(
                    path,
                    tree.use_tree_list(),
                    tree.rename(),
                    tree.star_token().is_some(),
                ),
            )
            .clone_for_update();

            tree.remove();
            ted::insert(Position::after(use_.syntax()), new_use.syntax());

            builder.replace(old_parent_range, new_parent.to_string());
        },
    )
}

fn resolve_full_path(tree: &ast::UseTree) -> Option<ast::Path> {
    let mut paths = tree
        .syntax()
        .ancestors()
        .take_while(|n| n.kind() != SyntaxKind::USE_KW)
        .filter_map(ast::UseTree::cast)
        .filter_map(|t| t.path());

    let mut final_path = paths.next()?;
    for path in paths {
        final_path = ast::make::path_concat(path, final_path)
    }
    Some(final_path)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn skip_single_use_item() {
        cov_mark::check!(skip_single_use_item);
        check_assist_not_applicable(
            unmerge_use,
            r"
use std::fmt::Debug$0;
",
        );
        check_assist_not_applicable(
            unmerge_use,
            r"
use std::fmt::{Debug$0};
",
        );
        check_assist_not_applicable(
            unmerge_use,
            r"
use std::fmt::Debug as Dbg$0;
",
        );
    }

    #[test]
    fn skip_single_glob_import() {
        check_assist_not_applicable(
            unmerge_use,
            r"
use std::fmt::*$0;
",
        );
    }

    #[test]
    fn unmerge_use_item() {
        check_assist(
            unmerge_use,
            r"
use std::fmt::{Debug, Display$0};
",
            r"
use std::fmt::{Debug};
use std::fmt::Display;
",
        );

        check_assist(
            unmerge_use,
            r"
use std::fmt::{Debug, format$0, Display};
",
            r"
use std::fmt::{Debug, Display};
use std::fmt::format;
",
        );
    }

    #[test]
    fn unmerge_glob_import() {
        check_assist(
            unmerge_use,
            r"
use std::fmt::{*$0, Display};
",
            r"
use std::fmt::{Display};
use std::fmt::*;
",
        );
    }

    #[test]
    fn unmerge_renamed_use_item() {
        check_assist(
            unmerge_use,
            r"
use std::fmt::{Debug, Display as Disp$0};
",
            r"
use std::fmt::{Debug};
use std::fmt::Display as Disp;
",
        );
    }

    #[test]
    fn unmerge_indented_use_item() {
        check_assist(
            unmerge_use,
            r"
mod format {
    use std::fmt::{Debug, Display$0 as Disp, format};
}
",
            r"
mod format {
    use std::fmt::{Debug, format};
    use std::fmt::Display as Disp;
}
",
        );
    }

    #[test]
    fn unmerge_nested_use_item() {
        check_assist(
            unmerge_use,
            r"
use foo::bar::{baz::{qux$0, foobar}, barbaz};
",
            r"
use foo::bar::{baz::{foobar}, barbaz};
use foo::bar::baz::qux;
",
        );
        check_assist(
            unmerge_use,
            r"
use foo::bar::{baz$0::{qux, foobar}, barbaz};
",
            r"
use foo::bar::{barbaz};
use foo::bar::baz::{qux, foobar};
",
        );
    }

    #[test]
    fn unmerge_use_item_with_visibility() {
        check_assist(
            unmerge_use,
            r"
pub use std::fmt::{Debug, Display$0};
",
            r"
pub use std::fmt::{Debug};
pub use std::fmt::Display;
",
        );
    }
}
