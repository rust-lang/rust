//! Settings for tweaking assists.
//!
//! The fun thing here is `SnippetCap` -- this type can only be created in this
//! module, and we use to statically check that we only produce snippet
//! assists if we are allowed to.

use hir::ImportPathConfig;
use ide_db::{SnippetCap, assists::ExprFillDefaultMode, imports::insert_use::InsertUseConfig};

use crate::AssistKind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssistConfig {
    pub snippet_cap: Option<SnippetCap>,
    pub allowed: Option<Vec<AssistKind>>,
    pub insert_use: InsertUseConfig,
    pub prefer_no_std: bool,
    pub prefer_prelude: bool,
    pub prefer_absolute: bool,
    pub assist_emit_must_use: bool,
    pub term_search_fuel: u64,
    pub term_search_borrowck: bool,
    pub code_action_grouping: bool,
    pub expr_fill_default: ExprFillDefaultMode,
    pub prefer_self_ty: bool,
}

impl AssistConfig {
    pub fn import_path_config(&self) -> ImportPathConfig {
        ImportPathConfig {
            prefer_no_std: self.prefer_no_std,
            prefer_prelude: self.prefer_prelude,
            prefer_absolute: self.prefer_absolute,
            allow_unstable: true,
        }
    }
}
