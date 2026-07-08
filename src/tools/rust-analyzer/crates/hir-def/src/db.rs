//! Defines database & queries for name resolution.
use base_db::SourceDatabase;

#[query_group::query_group]
pub trait DefDatabase: SourceDatabase {}
