//! `completions` crate provides utilities for generating completions of user input.

mod completions;
mod config;
mod context;
mod item;
mod render;

mod snippet;
#[cfg(test)]
mod tests;

use ide_db::{
    FilePosition, FxHashSet, RootDatabase,
    imports::insert_use::{self, ImportScope},
    syntax_helpers::tree_diff::diff,
    text_edit::TextEdit,
};
use syntax::ast::make;

use crate::{
    completions::Completions,
    context::{
        CompletionAnalysis, CompletionContext, NameRefContext, NameRefKind, PathCompletionCtx,
        PathKind,
    },
};

pub use crate::{
    config::{AutoImportExclusionType, CallableSnippets, CompletionConfig},
    item::{
        CompletionItem, CompletionItemKind, CompletionItemRefMode, CompletionRelevance,
        CompletionRelevancePostfixMatch, CompletionRelevanceReturnType,
        CompletionRelevanceTypeMatch,
    },
    snippet::{Snippet, SnippetScope},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CompletionFieldsToResolve {
    pub resolve_label_details: bool,
    pub resolve_tags: bool,
    pub resolve_detail: bool,
    pub resolve_documentation: bool,
    pub resolve_filter_text: bool,
    pub resolve_text_edit: bool,
    pub resolve_command: bool,
}

impl CompletionFieldsToResolve {
    pub fn from_client_capabilities(client_capability_fields: &FxHashSet<&str>) -> Self {
        Self {
            resolve_label_details: client_capability_fields.contains("labelDetails"),
            resolve_tags: client_capability_fields.contains("tags"),
            resolve_detail: client_capability_fields.contains("detail"),
            resolve_documentation: client_capability_fields.contains("documentation"),
            resolve_filter_text: client_capability_fields.contains("filterText"),
            resolve_text_edit: client_capability_fields.contains("textEdit"),
            resolve_command: client_capability_fields.contains("command"),
        }
    }

    pub const fn empty() -> Self {
        Self {
            resolve_label_details: false,
            resolve_tags: false,
            resolve_detail: false,
            resolve_documentation: false,
            resolve_filter_text: false,
            resolve_text_edit: false,
            resolve_command: false,
        }
    }
}

//FIXME: split the following feature into fine-grained features.

// Feature: Magic Completions
//
// In addition to usual reference completion, rust-analyzer provides some ✨magic✨
// completions as well:
//
// Keywords like `if`, `else` `while`, `loop` are completed with braces, and cursor
// is placed at the appropriate position. Even though `if` is easy to type, you
// still want to complete it, to get ` { }` for free! `return` is inserted with a
// space or `;` depending on the return type of the function.
//
// When completing a function call, `()` are automatically inserted. If a function
// takes arguments, the cursor is positioned inside the parenthesis.
//
// There are postfix completions, which can be triggered by typing something like
// `foo().if`. The word after `.` determines postfix completion. Possible variants are:
//
// - `expr.if` -> `if expr {}` or `if let ... {}` for `Option` or `Result`
// - `expr.match` -> `match expr {}`
// - `expr.while` -> `while expr {}` or `while let ... {}` for `Option` or `Result`
// - `expr.ref` -> `&expr`
// - `expr.refm` -> `&mut expr`
// - `expr.let` -> `let $0 = expr;`
// - `expr.lete` -> `let $1 = expr else { $0 };`
// - `expr.letm` -> `let mut $0 = expr;`
// - `expr.not` -> `!expr`
// - `expr.dbg` -> `dbg!(expr)`
// - `expr.dbgr` -> `dbg!(&expr)`
// - `expr.call` -> `(expr)`
//
// There also snippet completions:
//
// #### Expressions
//
// - `pd` -> `eprintln!(" = {:?}", );`
// - `ppd` -> `eprintln!(" = {:#?}", );`
//
// #### Items
//
// - `tfn` -> `#[test] fn feature(){}`
// - `tmod` ->
// ```rust
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_name() {}
// }
// ```
//
// And the auto import completions, enabled with the `rust-analyzer.completion.autoimport.enable` setting and the corresponding LSP client capabilities.
// Those are the additional completion options with automatic `use` import and options from all project importable items,
// fuzzy matched against the completion input.
//
// ![Magic Completions](https://user-images.githubusercontent.com/48062697/113020667-b72ab880-917a-11eb-8778-716cf26a0eb3.gif)

/// Main entry point for completion. We run completion as a two-phase process.
///
/// First, we look at the position and collect a so-called `CompletionContext`.
/// This is a somewhat messy process, because, during completion, syntax tree is
/// incomplete and can look really weird.
///
/// Once the context is collected, we run a series of completion routines which
/// look at the context and produce completion items. One subtlety about this
/// phase is that completion engine should not filter by the substring which is
/// already present, it should give all possible variants for the identifier at
/// the caret. In other words, for
///
/// ```ignore
/// fn f() {
///     let foo = 92;
///     let _ = bar$0
/// }
/// ```
///
/// `foo` *should* be present among the completion variants. Filtering by
/// identifier prefix/fuzzy match should be done higher in the stack, together
/// with ordering of completions (currently this is done by the client).
///
/// # Speculative Completion Problem
///
/// There's a curious unsolved problem in the current implementation. Often, you
/// want to compute completions on a *slightly different* text document.
///
/// In the simplest case, when the code looks like `let x = `, you want to
/// insert a fake identifier to get a better syntax tree: `let x = complete_me`.
///
/// We do this in `CompletionContext`, and it works OK-enough for *syntax*
/// analysis. However, we might want to, eg, ask for the type of `complete_me`
/// variable, and that's where our current infrastructure breaks down. salsa
/// doesn't allow such "phantom" inputs.
///
/// Another case where this would be instrumental is macro expansion. We want to
/// insert a fake ident and re-expand code. There's `expand_speculative` as a
/// workaround for this.
///
/// A different use-case is completion of injection (examples and links in doc
/// comments). When computing completion for a path in a doc-comment, you want
/// to inject a fake path expression into the item being documented and complete
/// that.
///
/// IntelliJ has CodeFragment/Context infrastructure for that. You can create a
/// temporary PSI node, and say that the context ("parent") of this node is some
/// existing node. Asking for, eg, type of this `CodeFragment` node works
/// correctly, as the underlying infrastructure makes use of contexts to do
/// analysis.
pub fn completions(
    db: &RootDatabase,
    config: &CompletionConfig<'_>,
    position: FilePosition,
    trigger_character: Option<char>,
) -> Option<Vec<CompletionItem>> {
    let (ctx, analysis) = &CompletionContext::new(db, position, config)?;
    let mut completions = Completions::default();

    // prevent `(` from triggering unwanted completion noise
    if trigger_character == Some('(') {
        if let CompletionAnalysis::NameRef(NameRefContext {
            kind:
                NameRefKind::Path(
                    path_ctx @ PathCompletionCtx { kind: PathKind::Vis { has_in_token }, .. },
                ),
            ..
        }) = analysis
        {
            completions::vis::complete_vis_path(&mut completions, ctx, path_ctx, has_in_token);
        }
        return Some(completions.into());
    }

    // when the user types a bare `_` (that is it does not belong to an identifier)
    // the user might just wanted to type a `_` for type inference or pattern discarding
    // so try to suppress completions in those cases
    if trigger_character == Some('_')
        && ctx.original_token.kind() == syntax::SyntaxKind::UNDERSCORE
        && let CompletionAnalysis::NameRef(NameRefContext {
            kind:
                NameRefKind::Path(
                    path_ctx @ PathCompletionCtx {
                        kind: PathKind::Type { .. } | PathKind::Pat { .. },
                        ..
                    },
                ),
            ..
        }) = analysis
        && path_ctx.is_trivial_path()
    {
        return None;
    }

    {
        let acc = &mut completions;

        match analysis {
            CompletionAnalysis::Name(name_ctx) => completions::complete_name(acc, ctx, name_ctx),
            CompletionAnalysis::NameRef(name_ref_ctx) => {
                completions::complete_name_ref(acc, ctx, name_ref_ctx)
            }
            CompletionAnalysis::Lifetime(lifetime_ctx) => {
                completions::lifetime::complete_label(acc, ctx, lifetime_ctx);
                completions::lifetime::complete_lifetime(acc, ctx, lifetime_ctx);
            }
            CompletionAnalysis::String { original, expanded: Some(expanded) } => {
                completions::extern_abi::complete_extern_abi(acc, ctx, expanded);
                completions::format_string::format_string(acc, ctx, original, expanded);
                completions::env_vars::complete_cargo_env_vars(acc, ctx, original, expanded);
            }
            CompletionAnalysis::UnexpandedAttrTT {
                colon_prefix,
                fake_attribute_under_caret: Some(attr),
                extern_crate,
            } => {
                completions::attribute::complete_known_attribute_input(
                    acc,
                    ctx,
                    colon_prefix,
                    attr,
                    extern_crate.as_ref(),
                );
            }
            CompletionAnalysis::UnexpandedAttrTT { .. } | CompletionAnalysis::String { .. } => (),
        }
    }

    Some(completions.into())
}

/// Resolves additional completion data at the position given.
/// This is used for import insertion done via completions like flyimport and custom user snippets.
pub fn resolve_completion_edits(
    db: &RootDatabase,
    config: &CompletionConfig<'_>,
    FilePosition { file_id, offset }: FilePosition,
    imports: impl IntoIterator<Item = String>,
) -> Option<Vec<TextEdit>> {
    let _p = tracing::info_span!("resolve_completion_edits").entered();
    let sema = hir::Semantics::new(db);

    let editioned_file_id = sema.attach_first_edition(file_id)?;

    let original_file = sema.parse(editioned_file_id);
    let original_token =
        syntax::AstNode::syntax(&original_file).token_at_offset(offset).left_biased()?;
    let position_for_import = &original_token.parent()?;
    let scope = ImportScope::find_insert_use_container(position_for_import, &sema)?;

    let current_module = sema.scope(position_for_import)?.module();
    let current_crate = current_module.krate();
    let current_edition = current_crate.edition(db);
    let new_ast = scope.clone_for_update();
    let mut import_insert = TextEdit::builder();

    imports.into_iter().for_each(|full_import_path| {
        insert_use::insert_use(
            &new_ast,
            make::path_from_text_with_edition(&full_import_path, current_edition),
            &config.insert_use,
        );
    });

    diff(scope.as_syntax_node(), new_ast.as_syntax_node()).into_text_edit(&mut import_insert);
    Some(vec![import_insert.finish()])
}
