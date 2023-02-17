use crate::errors::DumpVTableEntries;
use crate::traits::{impossible_predicates, is_vtable_safe_method};
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_infer::traits::util::PredicateSet;
use rustc_infer::traits::ImplSource;
use rustc_middle::ty::visit::TypeVisitable;
use rustc_middle::ty::InternalSubsts;
use rustc_middle::ty::{self, GenericParamDefKind, ToPredicate, Ty, TyCtxt, VtblEntry};
use rustc_span::{sym, Span};
use smallvec::SmallVec;

use std::fmt::Debug;
use std::ops::ControlFlow;

#[derive(Clone, Debug)]
pub(super) enum VtblSegment<'tcx> {
    MetadataDSA,
    TraitOwnEntries { trait_ref: ty::PolyTraitRef<'tcx>, emit_vptr: bool },
}

/// Prepare the segments for a vtable
pub(super) fn prepare_vtable_segments<'tcx, T>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::PolyTraitRef<'tcx>,
    mut segment_visitor: impl FnMut(VtblSegment<'tcx>) -> ControlFlow<T>,
) -> Option<T> {
    // The following constraints holds for the final arrangement.
    // 1. The whole virtual table of the first direct super trait is included as the
    //    the prefix. If this trait doesn't have any super traits, then this step
    //    consists of the dsa metadata.
    // 2. Then comes the proper pointer metadata(vptr) and all own methods for all
    //    other super traits except those already included as part of the first
    //    direct super trait virtual table.
    // 3. finally, the own methods of this trait.

    // This has the advantage that trait upcasting to the first direct super trait on each level
    // is zero cost, and to another trait includes only replacing the pointer with one level indirection,
    // while not using too much extra memory.

    // For a single inheritance relationship like this,
    //   D --> C --> B --> A
    // The resulting vtable will consists of these segments:
    //  DSA, A, B, C, D

    // For a multiple inheritance relationship like this,
    //   D --> C --> A
    //           \-> B
    // The resulting vtable will consists of these segments:
    //  DSA, A, B, B-vptr, C, D

    // For a diamond inheritance relationship like this,
    //   D --> B --> A
    //     \-> C -/
    // The resulting vtable will consists of these segments:
    //  DSA, A, B, C, C-vptr, D

    // For a more complex inheritance relationship like this:
    //   O --> G --> C --> A
    //     \     \     \-> B
    //     |     |-> F --> D
    //     |           \-> E
    //     |-> N --> J --> H
    //           \     \-> I
    //           |-> M --> K
    //                 \-> L
    // The resulting vtable will consists of these segments:
    //  DSA, A, B, B-vptr, C, D, D-vptr, E, E-vptr, F, F-vptr, G,
    //  H, H-vptr, I, I-vptr, J, J-vptr, K, K-vptr, L, L-vptr, M, M-vptr,
    //  N, N-vptr, O

    // emit dsa segment first.
    if let ControlFlow::Break(v) = (segment_visitor)(VtblSegment::MetadataDSA) {
        return Some(v);
    }

    let mut emit_vptr_on_new_entry = false;
    let mut visited = PredicateSet::new(tcx);
    let predicate = trait_ref.without_const().to_predicate(tcx);
    let mut stack: SmallVec<[(ty::PolyTraitRef<'tcx>, _, _); 5]> =
        smallvec![(trait_ref, emit_vptr_on_new_entry, None)];
    visited.insert(predicate);

    // the main traversal loop:
    // basically we want to cut the inheritance directed graph into a few non-overlapping slices of nodes
    // that each node is emitted after all its descendents have been emitted.
    // so we convert the directed graph into a tree by skipping all previously visited nodes using a visited set.
    // this is done on the fly.
    // Each loop run emits a slice - it starts by find a "childless" unvisited node, backtracking upwards, and it
    // stops after it finds a node that has a next-sibling node.
    // This next-sibling node will used as the starting point of next slice.

    // Example:
    // For a diamond inheritance relationship like this,
    //   D#1 --> B#0 --> A#0
    //     \-> C#1 -/

    // Starting point 0 stack [D]
    // Loop run #0: Stack after diving in is [D B A], A is "childless"
    // after this point, all newly visited nodes won't have a vtable that equals to a prefix of this one.
    // Loop run #0: Emitting the slice [B A] (in reverse order), B has a next-sibling node, so this slice stops here.
    // Loop run #0: Stack after exiting out is [D C], C is the next starting point.
    // Loop run #1: Stack after diving in is [D C], C is "childless", since its child A is skipped(already emitted).
    // Loop run #1: Emitting the slice [D C] (in reverse order). No one has a next-sibling node.
    // Loop run #1: Stack after exiting out is []. Now the function exits.

    loop {
        // dive deeper into the stack, recording the path
        'diving_in: loop {
            if let Some((inner_most_trait_ref, _, _)) = stack.last() {
                let inner_most_trait_ref = *inner_most_trait_ref;
                let mut direct_super_traits_iter = tcx
                    .super_predicates_of(inner_most_trait_ref.def_id())
                    .predicates
                    .into_iter()
                    .filter_map(move |(pred, _)| {
                        pred.subst_supertrait(tcx, &inner_most_trait_ref).to_opt_poly_trait_pred()
                    });

                'diving_in_skip_visited_traits: loop {
                    if let Some(next_super_trait) = direct_super_traits_iter.next() {
                        if visited.insert(next_super_trait.to_predicate(tcx)) {
                            // We're throwing away potential constness of super traits here.
                            // FIXME: handle ~const super traits
                            let next_super_trait = next_super_trait.map_bound(|t| t.trait_ref);
                            stack.push((
                                next_super_trait,
                                emit_vptr_on_new_entry,
                                Some(direct_super_traits_iter),
                            ));
                            break 'diving_in_skip_visited_traits;
                        } else {
                            continue 'diving_in_skip_visited_traits;
                        }
                    } else {
                        break 'diving_in;
                    }
                }
            }
        }

        // Other than the left-most path, vptr should be emitted for each trait.
        emit_vptr_on_new_entry = true;

        // emit innermost item, move to next sibling and stop there if possible, otherwise jump to outer level.
        'exiting_out: loop {
            if let Some((inner_most_trait_ref, emit_vptr, siblings_opt)) = stack.last_mut() {
                if let ControlFlow::Break(v) = (segment_visitor)(VtblSegment::TraitOwnEntries {
                    trait_ref: *inner_most_trait_ref,
                    emit_vptr: *emit_vptr,
                }) {
                    return Some(v);
                }

                'exiting_out_skip_visited_traits: loop {
                    if let Some(siblings) = siblings_opt {
                        if let Some(next_inner_most_trait_ref) = siblings.next() {
                            if visited.insert(next_inner_most_trait_ref.to_predicate(tcx)) {
                                // We're throwing away potential constness of super traits here.
                                // FIXME: handle ~const super traits
                                let next_inner_most_trait_ref =
                                    next_inner_most_trait_ref.map_bound(|t| t.trait_ref);
                                *inner_most_trait_ref = next_inner_most_trait_ref;
                                *emit_vptr = emit_vptr_on_new_entry;
                                break 'exiting_out;
                            } else {
                                continue 'exiting_out_skip_visited_traits;
                            }
                        }
                    }
                    stack.pop();
                    continue 'exiting_out;
                }
            }
            // all done
            return None;
        }
    }
}

fn dump_vtable_entries<'tcx>(
    tcx: TyCtxt<'tcx>,
    sp: Span,
    trait_ref: ty::PolyTraitRef<'tcx>,
    entries: &[VtblEntry<'tcx>],
) {
    tcx.sess.emit_err(DumpVTableEntries {
        span: sp,
        trait_ref,
        entries: format!("{:#?}", entries),
    });
}

fn own_existential_vtable_entries(tcx: TyCtxt<'_>, trait_def_id: DefId) -> &[DefId] {
    let trait_methods = tcx
        .associated_items(trait_def_id)
        .in_definition_order()
        .filter(|item| item.kind == ty::AssocKind::Fn);
    // Now list each method's DefId (for within its trait).
    let own_entries = trait_methods.filter_map(move |&trait_method| {
        debug!("own_existential_vtable_entry: trait_method={:?}", trait_method);
        let def_id = trait_method.def_id;

        // Some methods cannot be called on an object; skip those.
        if !is_vtable_safe_method(tcx, trait_def_id, trait_method) {
            debug!("own_existential_vtable_entry: not vtable safe");
            return None;
        }

        Some(def_id)
    });

    tcx.arena.alloc_from_iter(own_entries.into_iter())
}

/// Given a trait `trait_ref`, iterates the vtable entries
/// that come from `trait_ref`, including its supertraits.
fn vtable_entries<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::PolyTraitRef<'tcx>,
) -> &'tcx [VtblEntry<'tcx>] {
    debug!("vtable_entries({:?})", trait_ref);

    let mut entries = vec![];

    let vtable_segment_callback = |segment| -> ControlFlow<()> {
        match segment {
            VtblSegment::MetadataDSA => {
                entries.extend(TyCtxt::COMMON_VTABLE_ENTRIES);
            }
            VtblSegment::TraitOwnEntries { trait_ref, emit_vptr } => {
                let existential_trait_ref = trait_ref
                    .map_bound(|trait_ref| ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref));

                // Lookup the shape of vtable for the trait.
                let own_existential_entries =
                    tcx.own_existential_vtable_entries(existential_trait_ref.def_id());

                let own_entries = own_existential_entries.iter().copied().map(|def_id| {
                    debug!("vtable_entries: trait_method={:?}", def_id);

                    // The method may have some early-bound lifetimes; add regions for those.
                    let substs = trait_ref.map_bound(|trait_ref| {
                        InternalSubsts::for_item(tcx, def_id, |param, _| match param.kind {
                            GenericParamDefKind::Lifetime => tcx.lifetimes.re_erased.into(),
                            GenericParamDefKind::Type { .. }
                            | GenericParamDefKind::Const { .. } => {
                                trait_ref.substs[param.index as usize]
                            }
                        })
                    });

                    // The trait type may have higher-ranked lifetimes in it;
                    // erase them if they appear, so that we get the type
                    // at some particular call site.
                    let substs = tcx
                        .normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), substs);

                    // It's possible that the method relies on where-clauses that
                    // do not hold for this particular set of type parameters.
                    // Note that this method could then never be called, so we
                    // do not want to try and codegen it, in that case (see #23435).
                    let predicates = tcx.predicates_of(def_id).instantiate_own(tcx, substs);
                    if impossible_predicates(
                        tcx,
                        predicates.map(|(predicate, _)| predicate).collect(),
                    ) {
                        debug!("vtable_entries: predicates do not hold");
                        return VtblEntry::Vacant;
                    }

                    let instance = ty::Instance::resolve_for_vtable(
                        tcx,
                        ty::ParamEnv::reveal_all(),
                        def_id,
                        substs,
                    )
                    .expect("resolution failed during building vtable representation");
                    VtblEntry::Method(instance)
                });

                entries.extend(own_entries);

                if emit_vptr {
                    entries.push(VtblEntry::TraitVPtr(trait_ref));
                }
            }
        }

        ControlFlow::Continue(())
    };

    let _ = prepare_vtable_segments(tcx, trait_ref, vtable_segment_callback);

    if tcx.has_attr(trait_ref.def_id(), sym::rustc_dump_vtable) {
        let sp = tcx.def_span(trait_ref.def_id());
        dump_vtable_entries(tcx, sp, trait_ref, &entries);
    }

    tcx.arena.alloc_from_iter(entries.into_iter())
}

/// Find slot base for trait methods within vtable entries of another trait
pub(super) fn vtable_trait_first_method_offset<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (
        ty::PolyTraitRef<'tcx>, // trait_to_be_found
        ty::PolyTraitRef<'tcx>, // trait_owning_vtable
    ),
) -> usize {
    let (trait_to_be_found, trait_owning_vtable) = key;

    // #90177
    let trait_to_be_found_erased = tcx.erase_regions(trait_to_be_found);

    let vtable_segment_callback = {
        let mut vtable_base = 0;

        move |segment| {
            match segment {
                VtblSegment::MetadataDSA => {
                    vtable_base += TyCtxt::COMMON_VTABLE_ENTRIES.len();
                }
                VtblSegment::TraitOwnEntries { trait_ref, emit_vptr } => {
                    if tcx.erase_regions(trait_ref) == trait_to_be_found_erased {
                        return ControlFlow::Break(vtable_base);
                    }
                    vtable_base += count_own_vtable_entries(tcx, trait_ref);
                    if emit_vptr {
                        vtable_base += 1;
                    }
                }
            }
            ControlFlow::Continue(())
        }
    };

    if let Some(vtable_base) =
        prepare_vtable_segments(tcx, trait_owning_vtable, vtable_segment_callback)
    {
        vtable_base
    } else {
        bug!("Failed to find info for expected trait in vtable");
    }
}

/// Find slot offset for trait vptr within vtable entries of another trait
pub(crate) fn vtable_trait_upcasting_coercion_new_vptr_slot<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (
        Ty<'tcx>, // trait object type whose trait owning vtable
        Ty<'tcx>, // trait object for supertrait
    ),
) -> Option<usize> {
    let (source, target) = key;
    assert!(matches!(&source.kind(), &ty::Dynamic(..)) && !source.needs_infer());
    assert!(matches!(&target.kind(), &ty::Dynamic(..)) && !target.needs_infer());

    // this has been typecked-before, so diagnostics is not really needed.
    let unsize_trait_did = tcx.require_lang_item(LangItem::Unsize, None);

    let trait_ref = tcx.mk_trait_ref(unsize_trait_did, [source, target]);

    match tcx.codegen_select_candidate((ty::ParamEnv::reveal_all(), ty::Binder::dummy(trait_ref))) {
        Ok(ImplSource::TraitUpcasting(implsrc_traitcasting)) => {
            implsrc_traitcasting.vtable_vptr_slot
        }
        otherwise => bug!("expected TraitUpcasting candidate, got {otherwise:?}"),
    }
}

/// Given a trait `trait_ref`, returns the number of vtable entries
/// that come from `trait_ref`, excluding its supertraits. Used in
/// computing the vtable base for an upcast trait of a trait object.
pub(crate) fn count_own_vtable_entries<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::PolyTraitRef<'tcx>,
) -> usize {
    tcx.own_existential_vtable_entries(trait_ref.def_id()).len()
}

pub(super) fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        own_existential_vtable_entries,
        vtable_entries,
        vtable_trait_upcasting_coercion_new_vptr_slot,
        ..*providers
    };
}
