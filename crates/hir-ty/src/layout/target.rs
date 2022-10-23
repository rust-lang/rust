//! Target dependent parameters needed for layouts

use std::sync::Arc;

use hir_def::layout::TargetDataLayout;

use crate::db::HirDatabase;

use super::{AbiAndPrefAlign, AddressSpace, Align, Endian, Integer, Size};

pub fn current_target_data_layout_query(db: &dyn HirDatabase) -> Arc<TargetDataLayout> {
    let crate_graph = db.crate_graph();
    let cfg_options = &crate_graph[crate_graph.iter().next().unwrap()].cfg_options;
    let endian = match cfg_options.get_cfg_values("target_endian").next() {
        Some(x) if x.as_str() == "big" => Endian::Big,
        _ => Endian::Little,
    };
    let pointer_size =
        Size::from_bytes(match cfg_options.get_cfg_values("target_pointer_width").next() {
            Some(x) => match x.as_str() {
                "16" => 2,
                "32" => 4,
                _ => 8,
            },
            _ => 8,
        });
    Arc::new(TargetDataLayout {
        endian,
        i1_align: AbiAndPrefAlign::new(Align::from_bytes(1).unwrap()),
        i8_align: AbiAndPrefAlign::new(Align::from_bytes(1).unwrap()),
        i16_align: AbiAndPrefAlign::new(Align::from_bytes(2).unwrap()),
        i32_align: AbiAndPrefAlign::new(Align::from_bytes(4).unwrap()),
        i64_align: AbiAndPrefAlign::new(Align::from_bytes(8).unwrap()),
        i128_align: AbiAndPrefAlign::new(Align::from_bytes(8).unwrap()),
        f32_align: AbiAndPrefAlign::new(Align::from_bytes(4).unwrap()),
        f64_align: AbiAndPrefAlign::new(Align::from_bytes(8).unwrap()),
        pointer_size,
        pointer_align: AbiAndPrefAlign::new(Align::from_bytes(8).unwrap()),
        aggregate_align: AbiAndPrefAlign::new(Align::from_bytes(1).unwrap()),
        vector_align: vec![],
        instruction_address_space: AddressSpace(0),
        c_enum_min_size: Integer::I32,
    })
}
