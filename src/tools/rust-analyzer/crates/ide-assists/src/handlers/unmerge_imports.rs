use syntax::{
    AstNode, SyntaxKind,
    ast::{
        self, HasAttrs, HasVisibility, edit::IndentLevel, edit_in_place::AttrsOwnerEdit, make,
        syntax_factory::SyntaxFactory,
    },
    syntax_editor::{Element, Position, Removable},
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists},
};

// Assist: unmerge_imports
//
// Extracts a use item from a use list into a standalone use list.
//
// ```
// use std::fmt::{Debug, Display$0};
// ```
// ->
// ```
// use std::fmt::{Debug};
// use std::fmt::Display;
// ```
pub(crate) fn unmerge_imports(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let tree = ctx.find_node_at_offset::<ast::UseTree>()?;

    let tree_list = tree.syntax().parent().and_then(ast::UseTreeList::cast)?;
    if tree_list.use_trees().count() < 2 {
        cov_mark::hit!(skip_single_import);
        return None;
    }

    let use_ = tree_list.syntax().ancestors().find_map(ast::Use::cast)?;
    let path = resolve_full_path(&tree)?;

    // If possible, explain what is going to be done.
    let label = match tree.path().and_then(|path| path.first_segment()) {
        Some(name) => format!("Unmerge use of `{name}`"),
        None => "Unmerge use".into(),
    };

    let target = tree.syntax().text_range();
    acc.add(AssistId::refactor_rewrite("unmerge_imports"), label, target, |builder| {
        let make = SyntaxFactory::with_mappings();
        let new_use = make.use_(
            use_.visibility(),
            make.use_tree(path, tree.use_tree_list(), tree.rename(), tree.star_token().is_some()),
        );
        // Add any attributes that are present on the use tree
        use_.attrs().for_each(|attr| {
            new_use.add_attr(attr.clone_for_update());
        });

        let mut editor = builder.make_editor(use_.syntax());
        // Remove the use tree from the current use item
        tree.remove(&mut editor);
        // Insert a newline and indentation, followed by the new use item
        editor.insert_all(
            Position::after(use_.syntax()),
            vec![
                make.whitespace(&format!("\n{}", IndentLevel::from_node(use_.syntax())))
                    .syntax_element(),
                new_use.syntax().syntax_element(),
            ],
        );
        editor.add_mappings(make.finish_with_mappings());
        builder.add_file_edits(ctx.vfs_file_id(), editor);
    })
}

fn resolve_full_path(tree: &ast::UseTree) -> Option<ast::Path> {
    let paths = tree
        .syntax()
        .ancestors()
        .take_while(|n| n.kind() != SyntaxKind::USE)
        .filter_map(ast::UseTree::cast)
        .filter_map(|t| t.path());

    let final_path = paths.reduce(|prev, next| make::path_concat(next, prev))?;
    if final_path.segment().is_some_and(|it| it.self_token().is_some()) {
        final_path.qualifier()
    } else {
        Some(final_path)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn skip_single_import() {
        cov_mark::check!(skip_single_import);
        check_assist_not_applicable(
            unmerge_imports,
            r"
use std::fmt::Debug$0;
",
        );
        check_assist_not_applicable(
            unmerge_imports,
            r"
use std::fmt::{Debug$0};
",
        );
        check_assist_not_applicable(
            unmerge_imports,
            r"
use std::fmt::Debug as Dbg$0;
",
        );
    }

    #[test]
    fn skip_single_glob_import() {
        check_assist_not_applicable(
            unmerge_imports,
            r"
use std::fmt::*$0;
",
        );
    }

    #[test]
    fn unmerge_import() {
        check_assist(
            unmerge_imports,
            r"
use std::fmt::{Debug, Display$0};
",
            r"
use std::fmt::{Debug};
use std::fmt::Display;
",
        );

        check_assist(
            unmerge_imports,
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
            unmerge_imports,
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
    fn unmerge_renamed_import() {
        check_assist(
            unmerge_imports,
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
    fn unmerge_indented_import() {
        check_assist(
            unmerge_imports,
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
    fn unmerge_nested_import() {
        check_assist(
            unmerge_imports,
            r"
use foo::bar::{baz::{qux$0, foobar}, barbaz};
",
            r"
use foo::bar::{baz::{foobar}, barbaz};
use foo::bar::baz::qux;
",
        );
        check_assist(
            unmerge_imports,
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
    fn unmerge_import_with_visibility() {
        check_assist(
            unmerge_imports,
            r"
pub use std::fmt::{Debug, Display$0};
",
            r"
pub use std::fmt::{Debug};
pub use std::fmt::Display;
",
        );
    }

    #[test]
    fn unmerge_import_on_self() {
        check_assist(
            unmerge_imports,
            r"use std::process::{Command, self$0};",
            r"use std::process::{Command};
use std::process;",
        );
    }

    #[test]
    fn unmerge_import_with_attributes() {
        check_assist(
            unmerge_imports,
            r"
#[allow(deprecated)]
use foo::{bar, baz$0};",
            r"
#[allow(deprecated)]
use foo::{bar};
#[allow(deprecated)]
use foo::baz;",
        );
    }
}
