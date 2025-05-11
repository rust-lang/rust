//! Settings for tweaking completion.
//!
//! The fun thing here is `SnippetCap` -- this type can only be created in this
//! module, and we use to statically check that we only produce snippet
//! completions if we are allowed to.

use hir::ImportPathConfig;
use ide_db::{SnippetCap, imports::insert_use::InsertUseConfig};

use crate::{CompletionFieldsToResolve, snippet::Snippet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionConfig<'a> {
    pub enable_postfix_completions: bool,
    pub enable_imports_on_the_fly: bool,
    pub enable_self_on_the_fly: bool,
    pub enable_auto_iter: bool,
    pub enable_auto_await: bool,
    pub enable_private_editable: bool,
    pub enable_term_search: bool,
    pub term_search_fuel: u64,
    pub full_function_signatures: bool,
    pub callable: Option<CallableSnippets>,
    pub add_semicolon_to_unit: bool,
    pub snippet_cap: Option<SnippetCap>,
    pub insert_use: InsertUseConfig,
    pub prefer_no_std: bool,
    pub prefer_prelude: bool,
    pub prefer_absolute: bool,
    pub snippets: Vec<Snippet>,
    pub limit: Option<usize>,
    pub fields_to_resolve: CompletionFieldsToResolve,
    pub exclude_flyimport: Vec<(String, AutoImportExclusionType)>,
    pub exclude_traits: &'a [String],
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AutoImportExclusionType {
    Always,
    Methods,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallableSnippets {
    FillArguments,
    AddParentheses,
}

impl CompletionConfig<'_> {
    pub fn postfix_snippets(&self) -> impl Iterator<Item = (&str, &Snippet)> {
        self.snippets
            .iter()
            .flat_map(|snip| snip.postfix_triggers.iter().map(move |trigger| (&**trigger, snip)))
    }

    pub fn prefix_snippets(&self) -> impl Iterator<Item = (&str, &Snippet)> {
        self.snippets
            .iter()
            .flat_map(|snip| snip.prefix_triggers.iter().map(move |trigger| (&**trigger, snip)))
    }

    pub fn import_path_config(&self, allow_unstable: bool) -> ImportPathConfig {
        ImportPathConfig {
            prefer_no_std: self.prefer_no_std,
            prefer_prelude: self.prefer_prelude,
            prefer_absolute: self.prefer_absolute,
            allow_unstable,
        }
    }
}
