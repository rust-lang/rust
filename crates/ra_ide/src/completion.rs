//! FIXME: write short doc here

mod completion_item;
mod completion_context;
mod presentation;

mod complete_dot;
mod complete_record_literal;
mod complete_record_pattern;
mod complete_pattern;
mod complete_fn_param;
mod complete_keyword;
mod complete_snippet;
mod complete_path;
mod complete_scope;
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

pub use crate::completion::completion_item::{
    CompletionItem, CompletionItemKind, InsertTextFormat,
};
use either::Either;
use hir::{StructField, Type};
use ra_syntax::{
    ast::{self, NameOwner},
    SmolStr,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionConfig {
    pub enable_postfix_completions: bool,
    pub add_call_parenthesis: bool,
    pub add_call_argument_snippets: bool,
}

impl Default for CompletionConfig {
    fn default() -> Self {
        CompletionConfig {
            enable_postfix_completions: true,
            add_call_parenthesis: true,
            add_call_argument_snippets: true,
        }
    }
}

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
    position: FilePosition,
    config: &CompletionConfig,
) -> Option<Completions> {
    let ctx = CompletionContext::new(db, position, config)?;

    let mut acc = Completions::default();

    complete_fn_param::complete_fn_param(&mut acc, &ctx);
    complete_keyword::complete_expr_keyword(&mut acc, &ctx);
    complete_keyword::complete_use_tree_keyword(&mut acc, &ctx);
    complete_snippet::complete_expr_snippet(&mut acc, &ctx);
    complete_snippet::complete_item_snippet(&mut acc, &ctx);
    complete_path::complete_path(&mut acc, &ctx);
    complete_scope::complete_scope(&mut acc, &ctx);
    complete_dot::complete_dot(&mut acc, &ctx);
    complete_record_literal::complete_record_literal(&mut acc, &ctx);
    complete_record_pattern::complete_record_pattern(&mut acc, &ctx);
    complete_pattern::complete_pattern(&mut acc, &ctx);
    complete_postfix::complete_postfix(&mut acc, &ctx);
    complete_macro_in_item_position::complete_macro_in_item_position(&mut acc, &ctx);
    complete_trait_impl::complete_trait_impl(&mut acc, &ctx);

    Some(acc)
}

pub(crate) fn get_missing_fields(
    ctx: &CompletionContext,
    record: Either<&ast::RecordLit, &ast::RecordPat>,
) -> Option<Vec<(StructField, Type)>> {
    let (ty, variant) = match record {
        Either::Left(record_lit) => (
            ctx.sema.type_of_expr(&record_lit.clone().into())?,
            ctx.sema.resolve_record_literal(record_lit)?,
        ),
        Either::Right(record_pat) => (
            ctx.sema.type_of_pat(&record_pat.clone().into())?,
            ctx.sema.resolve_record_pattern(record_pat)?,
        ),
    };

    let already_present_names = get_already_present_names(record);
    Some(
        ty.variant_fields(ctx.db, variant)
            .into_iter()
            .filter(|(field, _)| {
                !already_present_names.contains(&SmolStr::from(field.name(ctx.db).to_string()))
            })
            .collect(),
    )
}

fn get_already_present_names(record: Either<&ast::RecordLit, &ast::RecordPat>) -> Vec<SmolStr> {
    // TODO kb have a single match
    match record {
        Either::Left(record_lit) => record_lit
            .record_field_list()
            .map(|field_list| field_list.fields())
            .map(|fields| {
                fields
                    .into_iter()
                    .filter_map(|field| field.name_ref())
                    .map(|name_ref| name_ref.text().clone())
                    .collect()
            })
            .unwrap_or_default(),
        Either::Right(record_pat) => record_pat
            .record_field_pat_list()
            .map(|pat_list| pat_list.bind_pats())
            .map(|bind_pats| {
                bind_pats
                    .into_iter()
                    .filter_map(|pat| pat.name())
                    .map(|name| name.text().clone())
                    .collect()
            })
            .unwrap_or_default(),
    }
}
