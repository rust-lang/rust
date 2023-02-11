//! Target dependent parameters needed for layouts

use std::sync::Arc;

use base_db::CrateId;
use hir_def::layout::{Endian, Size, TargetDataLayout};

use crate::db::HirDatabase;

pub fn target_data_layout_query(db: &dyn HirDatabase, krate: CrateId) -> Arc<TargetDataLayout> {
    let crate_graph = db.crate_graph();
    let target_layout = &crate_graph[krate].target_layout;
    let cfg_options = &crate_graph[krate].cfg_options;
    Arc::new(
        target_layout
            .as_ref()
            .and_then(|it| TargetDataLayout::parse_from_llvm_datalayout_string(it).ok())
            .unwrap_or_else(|| {
                let endian = match cfg_options.get_cfg_values("target_endian").next() {
                    Some(x) if x.as_str() == "big" => Endian::Big,
                    _ => Endian::Little,
                };
                let pointer_size = Size::from_bytes(
                    match cfg_options.get_cfg_values("target_pointer_width").next() {
                        Some(x) => match x.as_str() {
                            "16" => 2,
                            "32" => 4,
                            _ => 8,
                        },
                        _ => 8,
                    },
                );
                TargetDataLayout { endian, pointer_size, ..TargetDataLayout::default() }
            }),
    )
}
