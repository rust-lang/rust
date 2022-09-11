use ide_db::{base_db::FileId, source_change::SourceChange};
use itertools::Itertools;
use syntax::{ast, AstNode, SyntaxNode, TextRange};
use text_edit::TextEdit;

use crate::{fix, Diagnostic, Severity};

// Diagnostic: unnecessary-braces
//
// Diagnostic for unnecessary braces in `use` items.
pub(crate) fn useless_braces(
    acc: &mut Vec<Diagnostic>,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<()> {
    let use_tree_list = ast::UseTreeList::cast(node.clone())?;
    if let Some((single_use_tree,)) = use_tree_list.use_trees().collect_tuple() {
        // If there is a comment inside the bracketed `use`,
        // assume it is a commented out module path and don't show diagnostic.
        if use_tree_list.has_inner_comment() {
            return Some(());
        }

        let use_range = use_tree_list.syntax().text_range();
        let edit = remove_braces(&single_use_tree).unwrap_or_else(|| {
            let to_replace = single_use_tree.syntax().text().to_string();
            let mut edit_builder = TextEdit::builder();
            edit_builder.delete(use_range);
            edit_builder.insert(use_range.start(), to_replace);
            edit_builder.finish()
        });

        acc.push(
            Diagnostic::new(
                "unnecessary-braces",
                "Unnecessary braces in use statement".to_string(),
                use_range,
            )
            .severity(Severity::WeakWarning)
            .with_fixes(Some(vec![fix(
                "remove_braces",
                "Remove unnecessary braces",
                SourceChange::from_text_edit(file_id, edit),
                use_range,
            )])),
        );
    }

    Some(())
}

fn remove_braces(single_use_tree: &ast::UseTree) -> Option<TextEdit> {
    let use_tree_list_node = single_use_tree.syntax().parent()?;
    if single_use_tree.path()?.segment()?.self_token().is_some() {
        let start = use_tree_list_node.prev_sibling_or_token()?.text_range().start();
        let end = use_tree_list_node.text_range().end();
        return Some(TextEdit::delete(TextRange::new(start, end)));
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn test_check_unnecessary_braces_in_use_statement() {
        check_diagnostics(
            r#"
use a;
use a::{c, d::e};

mod a {
    mod c {}
    mod d {
        mod e {}
    }
}
"#,
        );
        check_diagnostics(
            r#"
use a;
use a::{
    c,
    // d::e
};

mod a {
    mod c {}
    mod d {
        mod e {}
    }
}
"#,
        );
        check_fix(
            r#"
mod b {}
use {$0b};
"#,
            r#"
mod b {}
use b;
"#,
        );
        check_fix(
            r#"
mod b {}
use {b$0};
"#,
            r#"
mod b {}
use b;
"#,
        );
        check_fix(
            r#"
mod a { mod c {} }
use a::{c$0};
"#,
            r#"
mod a { mod c {} }
use a::c;
"#,
        );
        check_fix(
            r#"
mod a {}
use a::{self$0};
"#,
            r#"
mod a {}
use a;
"#,
        );
        check_fix(
            r#"
mod a { mod c {} mod d { mod e {} } }
use a::{c, d::{e$0}};
"#,
            r#"
mod a { mod c {} mod d { mod e {} } }
use a::{c, d::e};
"#,
        );
    }
}
