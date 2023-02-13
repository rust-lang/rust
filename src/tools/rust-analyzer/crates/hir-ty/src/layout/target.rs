//! Target dependent parameters needed for layouts

use std::sync::Arc;

use base_db::CrateId;
use hir_def::layout::TargetDataLayout;

use crate::db::HirDatabase;

pub fn target_data_layout_query(
    db: &dyn HirDatabase,
    krate: CrateId,
) -> Option<Arc<TargetDataLayout>> {
    let crate_graph = db.crate_graph();
    let target_layout = crate_graph[krate].target_layout.as_ref().ok()?;
    let res = TargetDataLayout::parse_from_llvm_datalayout_string(&target_layout);
    if let Err(_e) = &res {
        // FIXME: Print the error here once it implements debug/display
        // also logging here is somewhat wrong, but unfortunately this is the earliest place we can
        // parse that doesn't impose a dependency to the rust-abi crate for project-model
        tracing::error!("Failed to parse target data layout for {krate:?}");
    }
    res.ok().map(Arc::new)
}
