mod completion_item;
mod completion_context;
mod presentation;

mod complete_dot;
mod complete_struct_literal;
mod complete_struct_pattern;
mod complete_pattern;
mod complete_fn_param;
mod complete_keyword;
mod complete_snippet;
mod complete_path;
mod complete_scope;
mod complete_postfix;

use ra_db::SourceDatabase;

#[cfg(test)]
use crate::completion::completion_item::do_completion;
use crate::{
    completion::{
        completion_context::CompletionContext,
        completion_item::{CompletionKind, Completions},
    },
    db, FilePosition,
};

pub use crate::completion::completion_item::{
    CompletionItem, CompletionItemKind, InsertTextFormat,
};

/// Main entry point for completion. We run completion as a two-phase process.
///
/// First, we look at the position and collect a so-called `CompletionContext.
/// This is a somewhat messy process, because, during completion, syntax tree is
/// incomplete and can look really weird.
///
/// Once the context is collected, we run a series of completion routines which
/// look at the context and produce completion items. One subtlety about this
/// phase is that completion engine should not filter by the substring which is
/// already present, it should give all possible variants for the identifier at
/// the caret. In other words, for
///
/// ```no-run
/// fn f() {
///     let foo = 92;
///     let _ = bar<|>
/// }
/// ```
///
/// `foo` *should* be present among the completion variants. Filtering by
/// identifier prefix/fuzzy match should be done higher in the stack, together
/// with ordering of completions (currently this is done by the client).
pub(crate) fn completions(db: &db::RootDatabase, position: FilePosition) -> Option<Completions> {
    let original_parse = db.parse(position.file_id);
    let ctx = CompletionContext::new(db, &original_parse, position)?;

    let mut acc = Completions::default();

    complete_fn_param::complete_fn_param(&mut acc, &ctx);
    complete_keyword::complete_expr_keyword(&mut acc, &ctx);
    complete_keyword::complete_use_tree_keyword(&mut acc, &ctx);
    complete_snippet::complete_expr_snippet(&mut acc, &ctx);
    complete_snippet::complete_item_snippet(&mut acc, &ctx);
    complete_path::complete_path(&mut acc, &ctx);
    complete_scope::complete_scope(&mut acc, &ctx);
    complete_dot::complete_dot(&mut acc, &ctx);
    complete_struct_literal::complete_struct_literal(&mut acc, &ctx);
    complete_struct_pattern::complete_struct_pattern(&mut acc, &ctx);
    complete_pattern::complete_pattern(&mut acc, &ctx);
    complete_postfix::complete_postfix(&mut acc, &ctx);
    Some(acc)
}
