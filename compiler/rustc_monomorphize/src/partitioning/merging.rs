use std::cmp;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::mir::mono::{CodegenUnit, CodegenUnitNameBuilder};
use rustc_span::symbol::Symbol;

use super::PartitioningCx;
use crate::partitioning::PreInliningPartitioning;

pub fn merge_codegen_units<'tcx>(
    cx: &PartitioningCx<'_, 'tcx>,
    initial_partitioning: &mut PreInliningPartitioning<'tcx>,
) {
    assert!(cx.target_cgu_count >= 1);
    let codegen_units = &mut initial_partitioning.codegen_units;

    // Note that at this point in time the `codegen_units` here may not be in a
    // deterministic order (but we know they're deterministically the same set).
    // We want this merging to produce a deterministic ordering of codegen units
    // from the input.
    //
    // Due to basically how we've implemented the merging below (merge the two
    // smallest into each other) we're sure to start off with a deterministic
    // order (sorted by name). This'll mean that if two cgus have the same size
    // the stable sort below will keep everything nice and deterministic.
    codegen_units.sort_by(|a, b| a.name().as_str().partial_cmp(b.name().as_str()).unwrap());

    // This map keeps track of what got merged into what.
    let mut cgu_contents: FxHashMap<Symbol, Vec<Symbol>> =
        codegen_units.iter().map(|cgu| (cgu.name(), vec![cgu.name()])).collect();

    // Merge the two smallest codegen units until the target size is reached.
    while codegen_units.len() > cx.target_cgu_count {
        // Sort small cgus to the back
        codegen_units.sort_by_cached_key(|cgu| cmp::Reverse(cgu.size_estimate()));
        let mut smallest = codegen_units.pop().unwrap();
        let second_smallest = codegen_units.last_mut().unwrap();

        // Move the mono-items from `smallest` to `second_smallest`
        second_smallest.modify_size_estimate(smallest.size_estimate());
        for (k, v) in smallest.items_mut().drain() {
            second_smallest.items_mut().insert(k, v);
        }

        // Record that `second_smallest` now contains all the stuff that was in
        // `smallest` before.
        let mut consumed_cgu_names = cgu_contents.remove(&smallest.name()).unwrap();
        cgu_contents.get_mut(&second_smallest.name()).unwrap().append(&mut consumed_cgu_names);

        debug!(
            "CodegenUnit {} merged into CodegenUnit {}",
            smallest.name(),
            second_smallest.name()
        );
    }

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(cx.tcx);

    if cx.tcx.sess.opts.incremental.is_some() {
        // If we are doing incremental compilation, we want CGU names to
        // reflect the path of the source level module they correspond to.
        // For CGUs that contain the code of multiple modules because of the
        // merging done above, we use a concatenation of the names of
        // all contained CGUs.
        let new_cgu_names: FxHashMap<Symbol, String> = cgu_contents
            .into_iter()
            // This `filter` makes sure we only update the name of CGUs that
            // were actually modified by merging.
            .filter(|(_, cgu_contents)| cgu_contents.len() > 1)
            .map(|(current_cgu_name, cgu_contents)| {
                let mut cgu_contents: Vec<&str> = cgu_contents.iter().map(|s| s.as_str()).collect();

                // Sort the names, so things are deterministic and easy to
                // predict.

                // We are sorting primitive &strs here so we can use unstable sort
                cgu_contents.sort_unstable();

                (current_cgu_name, cgu_contents.join("--"))
            })
            .collect();

        for cgu in codegen_units.iter_mut() {
            if let Some(new_cgu_name) = new_cgu_names.get(&cgu.name()) {
                if cx.tcx.sess.opts.debugging_opts.human_readable_cgu_names {
                    cgu.set_name(Symbol::intern(&new_cgu_name));
                } else {
                    // If we don't require CGU names to be human-readable, we
                    // use a fixed length hash of the composite CGU name
                    // instead.
                    let new_cgu_name = CodegenUnit::mangle_name(&new_cgu_name);
                    cgu.set_name(Symbol::intern(&new_cgu_name));
                }
            }
        }
    } else {
        // If we are compiling non-incrementally we just generate simple CGU
        // names containing an index.
        for (index, cgu) in codegen_units.iter_mut().enumerate() {
            cgu.set_name(numbered_codegen_unit_name(cgu_name_builder, index));
        }
    }
}

fn numbered_codegen_unit_name(
    name_builder: &mut CodegenUnitNameBuilder<'_>,
    index: usize,
) -> Symbol {
    name_builder.build_cgu_name_no_mangle(LOCAL_CRATE, &["cgu"], Some(index))
}
