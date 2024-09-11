//! Settings for tweaking completion.
//!
//! The fun thing here is `SnippetCap` -- this type can only be created in this
//! module, and we use to statically check that we only produce snippet
//! completions if we are allowed to.

use hir::ImportPathConfig;
use ide_db::{imports::insert_use::InsertUseConfig, SnippetCap};

use crate::snippet::Snippet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionConfig {
    pub enable_postfix_completions: bool,
    pub enable_imports_on_the_fly: bool,
    pub enable_self_on_the_fly: bool,
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallableSnippets {
    FillArguments,
    AddParentheses,
}

impl CompletionConfig {
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

    pub fn import_path_config(&self) -> ImportPathConfig {
        ImportPathConfig {
            prefer_no_std: self.prefer_no_std,
            prefer_prelude: self.prefer_prelude,
            prefer_absolute: self.prefer_absolute,
        }
    }
}
