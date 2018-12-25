mod completion_item;
mod completion_context;

mod complete_dot;
mod complete_fn_param;
mod complete_keyword;
mod complete_snippet;
mod complete_path;
mod complete_scope;

use ra_db::SyntaxDatabase;

use crate::{
    db,
    Cancelable, FilePosition,
    completion::{
        completion_item::{Completions, CompletionKind},
        completion_context::CompletionContext,
    },
};

pub use crate::completion::completion_item::{CompletionItem, InsertText, CompletionItemKind};

/// Main entry point for completion. We run completion as a two-phase process.
///
/// First, we look at the position and collect a so-called `CompletionContext.
/// This is a somewhat messy process, because, during completion, syntax tree is
/// incomplete and can look really weird.
///
/// Once the context is collected, we run a series of completion routines which
/// look at the context and produce completion items.
pub(crate) fn completions(
    db: &db::RootDatabase,
    position: FilePosition,
) -> Cancelable<Option<Completions>> {
    let original_file = db.source_file(position.file_id);
    let ctx = ctry!(CompletionContext::new(db, &original_file, position)?);

    let mut acc = Completions::default();

    complete_fn_param::complete_fn_param(&mut acc, &ctx);
    complete_keyword::complete_expr_keyword(&mut acc, &ctx);
    complete_snippet::complete_expr_snippet(&mut acc, &ctx);
    complete_snippet::complete_item_snippet(&mut acc, &ctx);
    complete_path::complete_path(&mut acc, &ctx)?;
    complete_scope::complete_scope(&mut acc, &ctx)?;
    complete_dot::complete_dot(&mut acc, &ctx)?;

    Ok(Some(acc))
}

#[cfg(test)]
fn check_completion(code: &str, expected_completions: &str, kind: CompletionKind) {
    use crate::mock_analysis::{single_file_with_position, analysis_and_position};
    let (analysis, position) = if code.contains("//-") {
        analysis_and_position(code)
    } else {
        single_file_with_position(code)
    };
    let completions = completions(&analysis.imp.db, position).unwrap().unwrap();
    completions.assert_match(expected_completions, kind);
}
