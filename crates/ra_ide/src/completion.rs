//! FIXME: write short doc here

mod completion_config;
mod completion_item;
mod completion_context;
mod presentation;

mod complete_attribute;
mod complete_dot;
mod complete_record;
mod complete_pattern;
mod complete_fn_param;
mod complete_keyword;
mod complete_snippet;
mod complete_qualified_path;
mod complete_unqualified_path;
mod complete_postfix;
mod complete_macro_in_item_position;
mod complete_trait_impl;
#[cfg(test)]
mod test_utils;

use ra_ide_db::RootDatabase;

use crate::{
    completion::{
        completion_context::CompletionContext,
        completion_item::{CompletionKind, Completions},
    },
    FilePosition,
};

pub use crate::completion::{
    completion_config::CompletionConfig,
    completion_item::{CompletionItem, CompletionItemKind, CompletionScore, InsertTextFormat},
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
pub(crate) fn completions(
    db: &RootDatabase,
    config: &CompletionConfig,
    position: FilePosition,
) -> Option<Completions> {
    let ctx = CompletionContext::new(db, position, config)?;

    let mut acc = Completions::default();
    complete_attribute::complete_attribute(&mut acc, &ctx);
    complete_fn_param::complete_fn_param(&mut acc, &ctx);
    complete_keyword::complete_expr_keyword(&mut acc, &ctx);
    complete_keyword::complete_use_tree_keyword(&mut acc, &ctx);
    complete_snippet::complete_expr_snippet(&mut acc, &ctx);
    complete_snippet::complete_item_snippet(&mut acc, &ctx);
    complete_qualified_path::complete_qualified_path(&mut acc, &ctx);
    complete_unqualified_path::complete_unqualified_path(&mut acc, &ctx);
    complete_dot::complete_dot(&mut acc, &ctx);
    complete_record::complete_record(&mut acc, &ctx);
    complete_pattern::complete_pattern(&mut acc, &ctx);
    complete_postfix::complete_postfix(&mut acc, &ctx);
    complete_macro_in_item_position::complete_macro_in_item_position(&mut acc, &ctx);
    complete_trait_impl::complete_trait_impl(&mut acc, &ctx);

    Some(acc)
}
