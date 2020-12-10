//! Settings for tweaking completion.
//!
//! The fun thing here is `SnippetCap` -- this type can only be created in this
//! module, and we use to statically check that we only produce snippet
//! completions if we are allowed to.

use ide_db::helpers::insert_use::MergeBehavior;
use rustc_hash::FxHashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionConfig {
    pub enable_postfix_completions: bool,
    pub enable_autoimport_completions: bool,
    pub add_call_parenthesis: bool,
    pub add_call_argument_snippets: bool,
    pub snippet_cap: Option<SnippetCap>,
    pub merge: Option<MergeBehavior>,
    /// A set of capabilities, enabled on the client and supported on the server.
    pub active_resolve_capabilities: FxHashSet<CompletionResolveCapability>,
}

/// A resolve capability, supported on the server.
/// If the client registers any completion resolve capabilities,
/// the server is able to render completion items' corresponding fields later,
/// not during an initial completion item request.
/// See https://github.com/rust-analyzer/rust-analyzer/issues/6366 for more details.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum CompletionResolveCapability {
    Documentation,
    Detail,
    AdditionalTextEdits,
}

impl CompletionConfig {
    pub fn allow_snippets(&mut self, yes: bool) {
        self.snippet_cap = if yes { Some(SnippetCap { _private: () }) } else { None }
    }

    /// Whether the completions' additional edits are calculated when sending an initional completions list
    /// or later, in a separate resolve request.
    pub fn resolve_additional_edits_lazily(&self) -> bool {
        self.active_resolve_capabilities.contains(&CompletionResolveCapability::AdditionalTextEdits)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnippetCap {
    _private: (),
}

impl Default for CompletionConfig {
    fn default() -> Self {
        CompletionConfig {
            enable_postfix_completions: true,
            enable_autoimport_completions: true,
            add_call_parenthesis: true,
            add_call_argument_snippets: true,
            snippet_cap: Some(SnippetCap { _private: () }),
            merge: Some(MergeBehavior::Full),
            active_resolve_capabilities: FxHashSet::default(),
        }
    }
}
