use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::mir::mono::{CodegenUnit, CodegenUnitNameBuilder};
use rustc_span::symbol::{Symbol, SymbolStr};

use super::PartitioningCx;
use crate::partitioning::PreInliningPartitioning;

use itertools::Itertools;
use itertools::MinMaxResult;

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
    codegen_units.sort_by_cached_key(|cgu| cgu.name().as_str());

    // This map keeps track of what got merged into what.
    let mut cgu_contents: FxHashMap<Symbol, Vec<SymbolStr>> =
        codegen_units.iter().map(|cgu| (cgu.name(), vec![cgu.name().as_str()])).collect();

    let mut set_of_subsets: Vec<Vec<Option<CodegenUnit<'_>>>> =
        Vec::with_capacity(initial_partitioning.codegen_units.len());
    for cgu in std::mem::take(&mut initial_partitioning.codegen_units) {
        let mut subsets = Vec::with_capacity(cx.target_cgu_count);
        subsets.push(Some(cgu));
        for _ in 1..cx.target_cgu_count {
            subsets.push(None);
        }
        set_of_subsets.push(subsets);
    }

    // From Wikipedia:
    // In each iteration, select two k-tuples A and B in which the difference
    // between the maximum and minimum sum is largest, and combine them in
    // reverse order of sizes, i.e.: smallest subset in A with largest subset in
    // B, second-smallest in A with second-largest in B, etc.
    while set_of_subsets.len() != 1 {
        set_of_subsets.sort_by_key(|subsets| {
            let mm =
                subsets.iter().minmax_by_key(|cgu| cgu.as_ref().map_or(0, |c| c.size_estimate()));
            match mm {
                MinMaxResult::NoElements => {
                    unreachable!(
                        "we always have at least one set with values in each set of subsets"
                    )
                }
                MinMaxResult::OneElement(a) => a.as_ref().map_or(0, |c| c.size_estimate()),
                MinMaxResult::MinMax(a, b) => {
                    b.as_ref().map_or(0, |c| c.size_estimate())
                        - a.as_ref().map_or(0, |c| c.size_estimate())
                }
            }
        });
        let mut a = set_of_subsets.pop().unwrap();
        let b = set_of_subsets.last_mut().unwrap();
        a.sort_by_key(|cgu| cgu.as_ref().map_or(0, |c| c.size_estimate()));
        b.sort_by_key(|cgu| cgu.as_ref().map_or(0, |c| c.size_estimate()));
        assert_eq!(a.len(), b.len(), "k-tuples");
        for (a, b) in a.into_iter().zip(b.iter_mut().rev()) {
            match (a, b) {
                (Some(mut a), Some(b)) => {
                    // Move the mono-items from `a` to `b`
                    b.modify_size_estimate(a.size_estimate());
                    for (k, v) in a.items_mut().drain() {
                        b.items_mut().insert(k, v);
                    }

                    // Record that `second_smallest` now contains all the stuff that was in
                    // `smallest` before.
                    let mut consumed_cgu_names = cgu_contents.remove(&a.name()).unwrap();
                    cgu_contents.get_mut(&b.name()).unwrap().append(&mut consumed_cgu_names);

                    debug!("CodegenUnit {} merged into CodegenUnit {}", a.name(), b.name());
                }
                (Some(a), b @ None) => {
                    *b = Some(a);
                }
                (None, Some(_)) => {
                    // no action needed to merge empty set into b.
                }
                (None, None) => {}
            }
        }
    }
    initial_partitioning.codegen_units =
        set_of_subsets.pop().unwrap().into_iter().filter_map(|v| v).collect::<Vec<_>>();
    let codegen_units = &mut initial_partitioning.codegen_units;
    assert!(codegen_units.len() <= cx.target_cgu_count);

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
                let mut cgu_contents: Vec<&str> = cgu_contents.iter().map(|s| &s[..]).collect();

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
