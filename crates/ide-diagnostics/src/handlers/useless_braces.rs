use ide_db::{base_db::FileId, source_change::SourceChange};
use itertools::Itertools;
use syntax::{ast, AstNode, SyntaxNode};
use text_edit::TextEdit;

use crate::{fix, Diagnostic, DiagnosticCode};

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
        // If there is a `self` inside the bracketed `use`, don't show diagnostic.
        if single_use_tree.path()?.segment()?.self_token().is_some() {
            return Some(());
        }

        // If there is a comment inside the bracketed `use`,
        // assume it is a commented out module path and don't show diagnostic.
        if use_tree_list.has_inner_comment() {
            return Some(());
        }

        let use_range = use_tree_list.syntax().text_range();
        let to_replace = single_use_tree.syntax().text().to_string();
        let mut edit_builder = TextEdit::builder();
        edit_builder.delete(use_range);
        edit_builder.insert(use_range.start(), to_replace);
        let edit = edit_builder.finish();

        acc.push(
            Diagnostic::new(
                DiagnosticCode::RustcLint("unused_braces"),
                "Unnecessary braces in use statement".to_string(),
                use_range,
            )
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

#[cfg(test)]
mod tests {
    use crate::{
        tests::{check_diagnostics, check_diagnostics_with_config, check_fix},
        DiagnosticsConfig,
    };

    #[test]
    fn test_check_unnecessary_braces_in_use_statement() {
        check_diagnostics(
            r#"
use a;
use a::{c, d::e};

mod a {
    pub mod c {}
    pub mod d {
        pub mod e {}
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
    pub mod c {}
    pub mod d {
        pub mod e {}
    }
}
"#,
        );
        check_diagnostics(
            r#"
use a::{self};

mod a {
}
"#,
        );
        check_diagnostics(
            r#"
use a::{self as cool_name};

mod a {
}
"#,
        );

        let mut config = DiagnosticsConfig::test_sample();
        config.disabled.insert("syntax-error".to_string());
        check_diagnostics_with_config(
            config,
            r#"
mod a { pub mod b {} }
use a::{b::self};
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
mod a { pub mod c {} }
use a::{c$0};
"#,
            r#"
mod a { pub mod c {} }
use a::c;
"#,
        );
        check_fix(
            r#"
mod a { pub mod c {} pub mod d { pub mod e {} } }
use a::{c, d::{e$0}};
"#,
            r#"
mod a { pub mod c {} pub mod d { pub mod e {} } }
use a::{c, d::e};
"#,
        );
    }
}
