//! Settings for tweaking assists.
//!
//! The fun thing here is `SnippetCap` -- this type can only be created in this
//! module, and we use to statically check that we only produce snippet
//! assists if we are allowed to.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssistConfig {
    pub snippet_cap: Option<SnippetCap>,
}

impl AssistConfig {
    pub fn allow_snippets(&mut self, yes: bool) {
        self.snippet_cap = if yes { Some(SnippetCap { _private: () }) } else { None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnippetCap {
    _private: (),
}

impl Default for AssistConfig {
    fn default() -> Self {
        AssistConfig { snippet_cap: Some(SnippetCap { _private: () }) }
    }
}
