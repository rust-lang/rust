//! Defines database & queries for macro expansion.

use base_db::SourceDatabase;

#[query_group::query_group]
pub trait ExpandDatabase: SourceDatabase {}
