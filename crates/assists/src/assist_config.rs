//! Settings for tweaking assists.
//!
//! The fun thing here is `SnippetCap` -- this type can only be created in this
//! module, and we use to statically check that we only produce snippet
//! assists if we are allowed to.

use hir::PrefixKind;
use ide_db::helpers::insert_use::MergeBehavior;

use crate::AssistKind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssistConfig {
    pub snippet_cap: Option<SnippetCap>,
    pub allowed: Option<Vec<AssistKind>>,
    pub insert_use: InsertUseConfig,
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
        AssistConfig {
            snippet_cap: Some(SnippetCap { _private: () }),
            allowed: None,
            insert_use: InsertUseConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InsertUseConfig {
    pub merge: Option<MergeBehavior>,
    pub prefix_kind: PrefixKind,
}

impl Default for InsertUseConfig {
    fn default() -> Self {
        InsertUseConfig { merge: Some(MergeBehavior::Full), prefix_kind: PrefixKind::Plain }
    }
}
