//! Re-exports various subcrates' databases and queries so that the calling code
//! can depend only on `hir`. This breaks abstraction boundary a bit, it would
//! be cool if we didn't do that.
//!
//! But we need this for at least LRU caching at the query level.
pub use hir_def::{
    db::{DefDatabase, set_expand_proc_attr_macros},
    file_item_tree,
};
pub use hir_ty::db::HirDatabase;
