//! Re-exports various subcrates databases so that the calling code can depend
//! only on `hir`. This breaks abstraction boundary a bit, it would be cool if
//! we didn't do that.
//!
//! But we need this for at least LRU caching at the query level.
pub use hir_def::db::*;
pub use hir_expand::db::{
    AstIdMapQuery, DeclMacroExpanderQuery, ExpandDatabase, ExpandDatabaseStorage,
    ExpandProcMacroQuery, HygieneFrameQuery, InternMacroCallQuery, MacroArgNodeQuery,
    MacroExpandQuery, ParseMacroExpansionErrorQuery, ParseMacroExpansionQuery,
};
pub use hir_ty::db::*;

#[test]
fn hir_database_is_object_safe() {
    fn _assert_object_safe(_: &dyn HirDatabase) {}
}
