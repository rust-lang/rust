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
    let target_layout = crate_graph[krate].target_layout.as_ref()?;
    Some(Arc::new(TargetDataLayout::parse_from_llvm_datalayout_string(&target_layout).ok()?))
}
