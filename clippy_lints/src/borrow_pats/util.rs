#![warn(unused)]
use std::collections::{BTreeMap, BTreeSet};

use clippy_utils::ty::{for_each_param_ty, for_each_ref_region, for_each_region};
use rustc_ast::Mutability;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::mir::{Body, Local, Operand, Place};
use rustc_middle::ty::{FnSig, GenericArgsRef, GenericPredicates, Region, Ty, TyCtxt, TyKind};
use rustc_span::source_map::Spanned;

use crate::borrow_pats::{LocalMagic, PlaceMagic};

mod visitor;
pub use visitor::*;

use super::prelude::RETURN_LOCAL;
use super::BodyStats;

const RETURN_RELEATION_INDEX: usize = usize::MAX;

pub struct PrintPrevent<T>(pub T);

impl<T> std::fmt::Debug for PrintPrevent<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PrintPrevent").finish()
    }
}

/// A helper struct to build relations between function arguments and the return
///
/// I really should stop using such stupid names. At this pooint I'm just making fun
/// of everything to make this work somehow tollerable.
#[derive(Debug)]
struct FuncReals<'tcx> {
    /// A list of several universes
    ///
    /// Mapping from `'short` (key) is outlives by `'long` (value)
    multiverse: BTreeMap<Region<'tcx>, BTreeSet<Region<'tcx>>>,
    sig: FnSig<'tcx>,
    args: GenericArgsRef<'tcx>,
    /// Indicates that a possibly returned value has generics with `'ReErased`
    has_generic_probles: bool,
}

impl<'tcx> FuncReals<'tcx> {
    fn from_fn_def(tcx: TyCtxt<'tcx>, def_id: DefId, args: GenericArgsRef<'tcx>) -> Self {
        // FIXME: The proper and long therm solution would be to use HIR
        // to find the call with generics that still have valid region markers.
        // However, for now I need to get this zombie in the air and not pefect
        let fn_sig = tcx.fn_sig(def_id).instantiate_identity();

        // On other functions this shouldn't matter. Even if they have late bounds
        // in their signature. We don't know how it's used and more imporantly,
        // The input and return types still need to follow Rust's type rules
        let fn_sig = fn_sig.skip_binder();

        let mut reals = Self {
            multiverse: Default::default(),
            sig: fn_sig,
            args,
            has_generic_probles: false,
        };

        // FYI: Predicates don't include transitive bounds
        let item_predicates = tcx.predicates_of(def_id);
        // TODO Test: `inferred_outlives_of`
        reals.build_multiverse(item_predicates);

        reals
    }

    fn build_multiverse(&mut self, predicates: GenericPredicates<'tcx>) {
        let preds = predicates
            .predicates
            .iter()
            .filter_map(|(clause, _span)| clause.as_region_outlives_clause());

        // I know this can be done in linear time, but I wasn't able to get this to
        // work quickly. So I'll do the n^2 version for now
        for binder in preds {
            // By now I believe (aka. wish) this is unimportant and can be removed.
            // But first I need to find something which actually triggers this todo.
            if !binder.bound_vars().is_empty() {
                todo!("Non empty depressing bounds 2: {binder:#?}");
            }

            let constaint = binder.skip_binder();
            let long = constaint.0;
            let short = constaint.1;

            let longi_verse = self.multiverse.get(&long).cloned().unwrap_or_default();
            let shorti_verse = self.multiverse.entry(short).or_default();
            if !shorti_verse.insert(long) {
                continue;
            }
            shorti_verse.extend(longi_verse);

            for universe in self.multiverse.values_mut() {
                if universe.contains(&short) {
                    universe.insert(long);
                }
            }
        }
    }

    fn relations(&mut self, dest: Local, args: &[Spanned<Operand<'tcx>>]) -> FxHashMap<Local, Vec<Local>> {
        let mut reals = FxHashMap::default();
        let ret_rels = self.return_relations();
        if !ret_rels.is_empty() {
            let locals: Vec<_> = ret_rels
                .into_iter()
                .filter_map(|idx| args[idx].node.place())
                .map(|place| place.local)
                .collect();
            if !locals.is_empty() {
                reals.insert(dest, locals);
            }
        }

        for (arg_index, arg_ty) in self.sig.inputs().iter().enumerate() {
            let mut arg_rels = FxHashSet::default();
            for_each_ref_region(*arg_ty, &mut |_reg, child_ty, mutability| {
                // `&X` is not really interesting here
                if matches!(mutability, Mutability::Mut) {
                    // This should be added here for composed types, like (&mut u32, &mut f32)
                    arg_rels.extend(self.find_relations(child_ty, arg_index));
                }
            });

            if !arg_rels.is_empty() {
                // It has to be a valid place, since we found a location
                let place = args[arg_index].node.place().unwrap();
                assert!(place.just_local());

                let locals: Vec<_> = arg_rels
                    .into_iter()
                    .filter_map(|idx| args[idx].node.place())
                    .map(|place| place.local)
                    .collect();
                if !locals.is_empty() {
                    reals.insert(place.local, locals);
                }
            }
        }

        reals
    }

    /// This function takes an operand, that identifies a function and returns the
    /// indices of the arguments that might be parents of the return type.
    ///
    /// ```
    /// fn example<'c, 'a: 'c, 'b: 'c>(cond: bool, a: &'a u32, b: &'b u32) -> &'c u32 {
    /// #    todo!()
    /// }
    /// ```
    /// This would return [1, 2], since the types in position 1 and 2 are related
    /// to the return type.
    fn return_relations(&mut self) -> FxHashSet<usize> {
        self.find_relations(self.sig.output(), RETURN_RELEATION_INDEX)
    }

    fn find_relations(&mut self, child_ty: Ty<'tcx>, child_index: usize) -> FxHashSet<usize> {
        let mut child_regions = FxHashSet::default();
        for_each_region(child_ty, |region| {
            if child_regions.insert(region) {
                if let Some(longer_regions) = self.multiverse.get(&region) {
                    child_regions.extend(longer_regions);
                }
            }
        });
        if child_index == RETURN_RELEATION_INDEX {
            for_each_param_ty(child_ty, &mut |param_ty| {
                if let Some(arg) = self.args.get(param_ty.index as usize) {
                    if let Some(arg_ty) = arg.as_type() {
                        for_each_region(arg_ty, |_| {
                            self.has_generic_probles = true;
                        });
                    }
                };
            });
        }

        let mut parents = FxHashSet::default();
        if child_regions.is_empty() {
            return parents;
        }

        for (index, ty) in self.sig.inputs().iter().enumerate() {
            if index == child_index {
                continue;
            }

            // "Here to stab things, don't case"
            for_each_ref_region(*ty, &mut |reg, _ty, _mutability| {
                if child_regions.contains(&reg) {
                    parents.insert(index);
                }
            });
        }

        parents
    }
}

pub fn calc_call_local_relations<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
    dest: Local,
    args: &[Spanned<Operand<'tcx>>],
    stats: &mut BodyStats,
) -> FxHashMap<Local, Vec<Local>> {
    let mut builder;
    if let Some((def_id, generic_args)) = func.const_fn_def() {
        builder = FuncReals::from_fn_def(tcx, def_id, generic_args);
    } else if let Some(place) = func.place() {
        let local_ty = body.local_decls[place.local].ty;
        if let TyKind::FnDef(def_id, generic_args) = local_ty.kind() {
            builder = FuncReals::from_fn_def(tcx, *def_id, generic_args);
        } else {
            stats.arg_relation_possibly_missed_due_to_late_bounds += 1;
            return FxHashMap::default();
        }
    } else {
        unreachable!()
    }
    let relations = builder.relations(dest, args);

    if builder.has_generic_probles {
        stats.arg_relation_possibly_missed_due_generics += 1;
    }

    relations
}

#[expect(clippy::needless_lifetimes)]
pub fn calc_fn_arg_relations<'tcx>(tcx: TyCtxt<'tcx>, fn_id: LocalDefId) -> FxHashMap<Local, Vec<Local>> {
    // This function is amazingly hacky, but at this point I really don't care anymore
    let mut builder = FuncReals::from_fn_def(tcx, fn_id.into(), rustc_middle::ty::List::empty());
    let arg_ctn = builder.sig.inputs().len();
    let fake_args: Vec<_> = (0..arg_ctn)
        .map(|idx| {
            // `_0` is the return, the arguments start at `_1`
            let place = Local::from_usize(idx + 1).as_place();
            let place = unsafe { std::mem::transmute::<Place<'static>, Place<'tcx>>(place) };
            Spanned {
                node: Operand::Move(place),
                span: rustc_span::DUMMY_SP,
            }
        })
        .collect();

    builder.relations(RETURN_LOCAL, &fake_args[..])
}

pub fn has_mut_ref(ty: Ty<'_>) -> bool {
    let mut has_mut = false;
    for_each_ref_region(ty, &mut |_reg, _ref_ty, mutability| {
        // `&X` is not really interesting here
        has_mut |= matches!(mutability, Mutability::Mut);
    });
    has_mut
}

/// Indicates the validity of a value.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum Validity {
    /// Is valid on all paths
    Valid,
    /// Maybe filled with valid data
    Maybe,
    /// Not filled with valid data
    Not,
}
