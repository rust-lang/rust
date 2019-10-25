use crate::hir::def_id::{DefId, DefIndex};
use crate::hir::{self, HirId};
use rustc_data_structures::sync::{Lock, ParallelIterator, par_iter};
use std::collections::BTreeMap;

pub fn check_crate(hir_map: &hir::map::Map<'_>) {
    hir_map.dep_graph.assert_ignored();

    let errors = Lock::new(Vec::new());

    par_iter(0..hir_map.map.len()).for_each(|owner| {
        let owner = DefIndex::from(owner);
        let local_map = &hir_map.map[owner];

        // Collect the missing `ItemLocalId`s.
        let missing: Vec<_> = local_map
            .iter_enumerated()
            .filter(|(_, entry)| entry.is_none())
            .map(|(local_id, _)| local_id)
            .collect();

        if !missing.is_empty() {
            let present: BTreeMap<_, _> = local_map
                .iter_enumerated()
                .filter(|(_, entry)| entry.is_some())
                .map(|(local_id, _)| {
                    (local_id, hir_map.node_to_string(HirId { owner, local_id }))
                })
                .collect();

            errors.lock().push(format!(
                "{}:\n  missing IDs = {:?}\n  present IDs = {:#?}",
                hir_map.def_path(DefId::local(owner)).to_string_no_crate(),
                missing,
                present,
            ));
        }
    });

    let errors = errors.into_inner();
    if !errors.is_empty() {
        bug!("`ItemLocalId`s not assigned densely in:\n{}", errors.join("\n"));
    }
}
