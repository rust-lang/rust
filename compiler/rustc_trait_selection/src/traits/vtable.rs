use std::fmt::Debug;
use std::ops::ControlFlow;

use rustc_hir::def_id::DefId;
use rustc_infer::traits::util::PredicateSet;
use rustc_middle::bug;
use rustc_middle::query::Providers;
use rustc_middle::ty::{
    self, GenericArgs, GenericParamDefKind, Ty, TyCtxt, TypeVisitableExt, Upcast, VtblEntry,
};
use rustc_span::DUMMY_SP;
use smallvec::{SmallVec, smallvec};
use tracing::debug;

use crate::traits::{impossible_predicates, is_vtable_safe_method};

#[derive(Clone, Debug)]
pub enum VtblSegment<'tcx> {
    MetadataDSA,
    TraitOwnEntries { trait_ref: ty::TraitRef<'tcx>, emit_vptr: bool },
}

/// Prepare the segments for a vtable
// FIXME: This should take a `PolyExistentialTraitRef`, since we don't care
// about our `Self` type here.
pub fn prepare_vtable_segments<'tcx, T>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    segment_visitor: impl FnMut(VtblSegment<'tcx>) -> ControlFlow<T>,
) -> Option<T> {
    prepare_vtable_segments_inner(tcx, trait_ref, segment_visitor).break_value()
}

/// Helper for [`prepare_vtable_segments`] that returns `ControlFlow`,
/// such that we can use `?` in the body.
fn prepare_vtable_segments_inner<'tcx, T>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    mut segment_visitor: impl FnMut(VtblSegment<'tcx>) -> ControlFlow<T>,
) -> ControlFlow<T> {
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
    segment_visitor(VtblSegment::MetadataDSA)?;

    let mut emit_vptr_on_new_entry = false;
    let mut visited = PredicateSet::new(tcx);
    let predicate = trait_ref.upcast(tcx);
    let mut stack: SmallVec<[(ty::TraitRef<'tcx>, _, _); 5]> =
        smallvec![(trait_ref, emit_vptr_on_new_entry, maybe_iter(None))];
    visited.insert(predicate);

    // the main traversal loop:
    // basically we want to cut the inheritance directed graph into a few non-overlapping slices of nodes
    // such that each node is emitted after all its descendants have been emitted.
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

    'outer: loop {
        // dive deeper into the stack, recording the path
        'diving_in: loop {
            let &(inner_most_trait_ref, _, _) = stack.last().unwrap();

            let mut direct_super_traits_iter = tcx
                .explicit_super_predicates_of(inner_most_trait_ref.def_id)
                .iter_identity_copied()
                .filter_map(move |(pred, _)| {
                    pred.instantiate_supertrait(tcx, ty::Binder::dummy(inner_most_trait_ref))
                        .as_trait_clause()
                })
                .map(move |pred| {
                    tcx.normalize_erasing_late_bound_regions(
                        ty::TypingEnv::fully_monomorphized(),
                        pred,
                    )
                    .trait_ref
                });

            // Find an unvisited supertrait
            match direct_super_traits_iter
                .find(|&super_trait| visited.insert(super_trait.upcast(tcx)))
            {
                // Push it to the stack for the next iteration of 'diving_in to pick up
                Some(next_super_trait) => stack.push((
                    next_super_trait,
                    emit_vptr_on_new_entry,
                    maybe_iter(Some(direct_super_traits_iter)),
                )),

                // There are no more unvisited direct super traits, dive-in finished
                None => break 'diving_in,
            }
        }

        // emit innermost item, move to next sibling and stop there if possible, otherwise jump to outer level.
        while let Some((inner_most_trait_ref, emit_vptr, mut siblings)) = stack.pop() {
            // We don't need to emit a vptr for "truly-empty" supertraits, but we *do* need to emit a
            // vptr for supertraits that have no methods, but that themselves have supertraits
            // with methods, so we check if any transitive supertrait has entries here (this includes
            // the trait itself).
            let has_entries = ty::elaborate::supertrait_def_ids(tcx, inner_most_trait_ref.def_id)
                .any(|def_id| has_own_existential_vtable_entries(tcx, def_id));

            segment_visitor(VtblSegment::TraitOwnEntries {
                trait_ref: inner_most_trait_ref,
                emit_vptr: emit_vptr && has_entries && !tcx.sess.opts.unstable_opts.no_trait_vptr,
            })?;

            // If we've emitted (fed to `segment_visitor`) a trait that has methods present in the vtable,
            // we'll need to emit vptrs from now on.
            emit_vptr_on_new_entry |= has_entries;

            if let Some(next_inner_most_trait_ref) =
                siblings.find(|&sibling| visited.insert(sibling.upcast(tcx)))
            {
                stack.push((next_inner_most_trait_ref, emit_vptr_on_new_entry, siblings));

                // just pushed a new trait onto the stack, so we need to go through its super traits
                continue 'outer;
            }
        }

        // the stack is empty, all done
        return ControlFlow::Continue(());
    }
}

/// Turns option of iterator into an iterator (this is just flatten)
fn maybe_iter<I: Iterator>(i: Option<I>) -> impl Iterator<Item = I::Item> {
    // Flatten is bad perf-vise, we could probably implement a special case here that is better
    i.into_iter().flatten()
}

fn has_own_existential_vtable_entries(tcx: TyCtxt<'_>, trait_def_id: DefId) -> bool {
    own_existential_vtable_entries_iter(tcx, trait_def_id).next().is_some()
}

fn own_existential_vtable_entries(tcx: TyCtxt<'_>, trait_def_id: DefId) -> &[DefId] {
    tcx.arena.alloc_from_iter(own_existential_vtable_entries_iter(tcx, trait_def_id))
}

fn own_existential_vtable_entries_iter(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
) -> impl Iterator<Item = DefId> {
    let trait_methods =
        tcx.associated_items(trait_def_id).in_definition_order().filter(|item| item.is_fn());

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

    own_entries
}

/// Given a trait `trait_ref`, iterates the vtable entries
/// that come from `trait_ref`, including its supertraits.
fn vtable_entries<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
) -> &'tcx [VtblEntry<'tcx>] {
    debug_assert!(!trait_ref.has_non_region_infer() && !trait_ref.has_non_region_param());
    debug_assert_eq!(
        tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), trait_ref),
        trait_ref,
        "vtable trait ref should be normalized"
    );

    debug!("vtable_entries({:?})", trait_ref);

    let mut entries = vec![];

    let vtable_segment_callback = |segment| -> ControlFlow<()> {
        match segment {
            VtblSegment::MetadataDSA => {
                entries.extend(TyCtxt::COMMON_VTABLE_ENTRIES);
            }
            VtblSegment::TraitOwnEntries { trait_ref, emit_vptr } => {
                let existential_trait_ref = ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref);

                // Lookup the shape of vtable for the trait.
                let own_existential_entries =
                    tcx.own_existential_vtable_entries(existential_trait_ref.def_id);

                let own_entries = own_existential_entries.iter().copied().map(|def_id| {
                    debug!("vtable_entries: trait_method={:?}", def_id);

                    // The method may have some early-bound lifetimes; add regions for those.
                    // FIXME: Is this normalize needed?
                    let args = tcx.normalize_erasing_regions(
                        ty::TypingEnv::fully_monomorphized(),
                        GenericArgs::for_item(tcx, def_id, |param, _| match param.kind {
                            GenericParamDefKind::Lifetime => tcx.lifetimes.re_erased.into(),
                            GenericParamDefKind::Type { .. }
                            | GenericParamDefKind::Const { .. } => {
                                trait_ref.args[param.index as usize]
                            }
                        }),
                    );

                    // It's possible that the method relies on where-clauses that
                    // do not hold for this particular set of type parameters.
                    // Note that this method could then never be called, so we
                    // do not want to try and codegen it, in that case (see #23435).
                    let predicates = tcx.predicates_of(def_id).instantiate_own(tcx, args);
                    if impossible_predicates(
                        tcx,
                        predicates.map(|(predicate, _)| predicate).collect(),
                    ) {
                        debug!("vtable_entries: predicates do not hold");
                        return VtblEntry::Vacant;
                    }

                    let instance = ty::Instance::expect_resolve_for_vtable(
                        tcx,
                        ty::TypingEnv::fully_monomorphized(),
                        def_id,
                        args,
                        DUMMY_SP,
                    );

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

    tcx.arena.alloc_from_iter(entries)
}

// Given a `dyn Subtrait: Supertrait` trait ref, find corresponding first slot
// for `Supertrait`'s methods in the vtable of `Subtrait`.
pub(crate) fn first_method_vtable_slot<'tcx>(tcx: TyCtxt<'tcx>, key: ty::TraitRef<'tcx>) -> usize {
    debug_assert!(!key.has_non_region_infer() && !key.has_non_region_param());
    debug_assert_eq!(
        tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), key),
        key,
        "vtable trait ref should be normalized"
    );

    let ty::Dynamic(source, _) = *key.self_ty().kind() else {
        bug!();
    };
    let source_principal = tcx.instantiate_bound_regions_with_erased(
        source.principal().unwrap().with_self_ty(tcx, key.self_ty()),
    );

    // We're monomorphizing a call to a dyn trait object that can never be constructed.
    if tcx.instantiate_and_check_impossible_predicates((
        source_principal.def_id,
        source_principal.args,
    )) {
        return 0;
    }

    let target_principal = ty::ExistentialTraitRef::erase_self_ty(tcx, key);

    let vtable_segment_callback = {
        let mut vptr_offset = 0;
        move |segment| {
            match segment {
                VtblSegment::MetadataDSA => {
                    vptr_offset += TyCtxt::COMMON_VTABLE_ENTRIES.len();
                }
                VtblSegment::TraitOwnEntries { trait_ref: vtable_principal, emit_vptr } => {
                    if ty::ExistentialTraitRef::erase_self_ty(tcx, vtable_principal)
                        == target_principal
                    {
                        return ControlFlow::Break(vptr_offset);
                    }

                    vptr_offset +=
                        tcx.own_existential_vtable_entries(vtable_principal.def_id).len();

                    if emit_vptr {
                        vptr_offset += 1;
                    }
                }
            }
            ControlFlow::Continue(())
        }
    };

    prepare_vtable_segments(tcx, source_principal, vtable_segment_callback).unwrap()
}

/// Given a `dyn Subtrait` and `dyn Supertrait` trait object, find the slot of
/// the trait vptr in the subtrait's vtable.
///
/// A return value of `None` means that the original vtable can be reused.
pub(crate) fn supertrait_vtable_slot<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (
        Ty<'tcx>, // Source -- `dyn Subtrait`.
        Ty<'tcx>, // Target -- `dyn Supertrait` being coerced to.
    ),
) -> Option<usize> {
    debug_assert!(!key.has_non_region_infer() && !key.has_non_region_param());
    debug_assert_eq!(
        tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), key),
        key,
        "upcasting trait refs should be normalized"
    );

    let (source, target) = key;

    // If the target principal is `None`, we can just return `None`.
    let ty::Dynamic(target_data, _) = *target.kind() else {
        bug!();
    };
    let target_principal = tcx.instantiate_bound_regions_with_erased(target_data.principal()?);

    // Given that we have a target principal, it is a bug for there not to be a source principal.
    let ty::Dynamic(source_data, _) = *source.kind() else {
        bug!();
    };
    let source_principal = tcx.instantiate_bound_regions_with_erased(
        source_data.principal().unwrap().with_self_ty(tcx, source),
    );

    // We're monomorphizing a dyn trait object upcast that can never be constructed.
    if tcx.instantiate_and_check_impossible_predicates((
        source_principal.def_id,
        source_principal.args,
    )) {
        return None;
    }

    let vtable_segment_callback = {
        let mut vptr_offset = 0;
        move |segment| {
            match segment {
                VtblSegment::MetadataDSA => {
                    vptr_offset += TyCtxt::COMMON_VTABLE_ENTRIES.len();
                }
                VtblSegment::TraitOwnEntries { trait_ref: vtable_principal, emit_vptr } => {
                    vptr_offset +=
                        tcx.own_existential_vtable_entries(vtable_principal.def_id).len();
                    if ty::ExistentialTraitRef::erase_self_ty(tcx, vtable_principal)
                        == target_principal
                    {
                        if emit_vptr {
                            return ControlFlow::Break(Some(vptr_offset));
                        } else {
                            return ControlFlow::Break(None);
                        }
                    }

                    if emit_vptr {
                        vptr_offset += 1;
                    }
                }
            }
            ControlFlow::Continue(())
        }
    };

    prepare_vtable_segments(tcx, source_principal, vtable_segment_callback).unwrap()
}

pub(super) fn provide(providers: &mut Providers) {
    *providers = Providers {
        own_existential_vtable_entries,
        vtable_entries,
        first_method_vtable_slot,
        supertrait_vtable_slot,
        ..*providers
    };
}
