// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::infer::canonical::{Canonical, QueryResult};
use rustc::hir::def_id::DefId;
use rustc::traits::{FulfillmentContext, Normalized, ObligationCause};
use rustc::traits::query::{CanonicalTyGoal, NoSolution};
use rustc::traits::query::dropck_outlives::{DtorckConstraint, DropckOutlivesResult};
use rustc::ty::{self, ParamEnvAnd, Ty, TyCtxt};
use rustc::ty::subst::Subst;
use rustc::util::nodemap::FxHashSet;
use rustc_data_structures::sync::Lrc;
use syntax::codemap::{Span, DUMMY_SP};
use util;

crate fn dropck_outlives<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    goal: CanonicalTyGoal<'tcx>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, DropckOutlivesResult<'tcx>>>>, NoSolution> {
    debug!("dropck_outlives(goal={:#?})", goal);

    tcx.infer_ctxt().enter(|ref infcx| {
        let tcx = infcx.tcx;
        let (
            ParamEnvAnd {
                param_env,
                value: for_ty,
            },
            canonical_inference_vars,
        ) = infcx.instantiate_canonical_with_fresh_inference_vars(DUMMY_SP, &goal);

        let mut result = DropckOutlivesResult { kinds: vec![], overflows: vec![] };

        // A stack of types left to process. Each round, we pop
        // something from the stack and invoke
        // `dtorck_constraint_for_ty`. This may produce new types that
        // have to be pushed on the stack. This continues until we have explored
        // all the reachable types from the type `for_ty`.
        //
        // Example: Imagine that we have the following code:
        //
        // ```rust
        // struct A {
        //     value: B,
        //     children: Vec<A>,
        // }
        //
        // struct B {
        //     value: u32
        // }
        //
        // fn f() {
        //   let a: A = ...;
        //   ..
        // } // here, `a` is dropped
        // ```
        //
        // at the point where `a` is dropped, we need to figure out
        // which types inside of `a` contain region data that may be
        // accessed by any destructors in `a`. We begin by pushing `A`
        // onto the stack, as that is the type of `a`. We will then
        // invoke `dtorck_constraint_for_ty` which will expand `A`
        // into the types of its fields `(B, Vec<A>)`. These will get
        // pushed onto the stack. Eventually, expanding `Vec<A>` will
        // lead to us trying to push `A` a second time -- to prevent
        // infinite recusion, we notice that `A` was already pushed
        // once and stop.
        let mut ty_stack = vec![(for_ty, 0)];

        // Set used to detect infinite recursion.
        let mut ty_set = FxHashSet();

        let fulfill_cx = &mut FulfillmentContext::new();

        let cause = ObligationCause::dummy();
        while let Some((ty, depth)) = ty_stack.pop() {
            let DtorckConstraint {
                dtorck_types,
                outlives,
                overflows,
            } = dtorck_constraint_for_ty(tcx, DUMMY_SP, for_ty, depth, ty)?;

            // "outlives" represent types/regions that may be touched
            // by a destructor.
            result.kinds.extend(outlives);
            result.overflows.extend(overflows);

            // dtorck types are "types that will get dropped but which
            // do not themselves define a destructor", more or less. We have
            // to push them onto the stack to be expanded.
            for ty in dtorck_types {
                match infcx.at(&cause, param_env).normalize(&ty) {
                    Ok(Normalized {
                        value: ty,
                        obligations,
                    }) => {
                        fulfill_cx.register_predicate_obligations(infcx, obligations);

                        debug!("dropck_outlives: ty from dtorck_types = {:?}", ty);

                        match ty.sty {
                            // All parameters live for the duration of the
                            // function.
                            ty::TyParam(..) => {}

                            // A projection that we couldn't resolve - it
                            // might have a destructor.
                            ty::TyProjection(..) | ty::TyAnon(..) => {
                                result.kinds.push(ty.into());
                            }

                            _ => {
                                if ty_set.insert(ty) {
                                    ty_stack.push((ty, depth + 1));
                                }
                            }
                        }
                    }

                    // We don't actually expect to fail to normalize.
                    // That implies a WF error somewhere else.
                    Err(NoSolution) => {
                        return Err(NoSolution);
                    }
                }
            }
        }

        debug!("dropck_outlives: result = {:#?}", result);

        util::make_query_response(infcx, canonical_inference_vars, result, fulfill_cx)
    })
}

/// Return a set of constraints that needs to be satisfied in
/// order for `ty` to be valid for destruction.
fn dtorck_constraint_for_ty<'a, 'gcx, 'tcx>(
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    span: Span,
    for_ty: Ty<'tcx>,
    depth: usize,
    ty: Ty<'tcx>,
) -> Result<DtorckConstraint<'tcx>, NoSolution> {
    debug!(
        "dtorck_constraint_for_ty({:?}, {:?}, {:?}, {:?})",
        span, for_ty, depth, ty
    );

    if depth >= *tcx.sess.recursion_limit.get() {
        return Ok(DtorckConstraint {
            outlives: vec![],
            dtorck_types: vec![],
            overflows: vec![ty],
        });
    }

    let result = match ty.sty {
        ty::TyBool
        | ty::TyChar
        | ty::TyInt(_)
        | ty::TyUint(_)
        | ty::TyFloat(_)
        | ty::TyStr
        | ty::TyNever
        | ty::TyForeign(..)
        | ty::TyRawPtr(..)
        | ty::TyRef(..)
        | ty::TyFnDef(..)
        | ty::TyFnPtr(_)
        | ty::TyGeneratorWitness(..) => {
            // these types never have a destructor
            Ok(DtorckConstraint::empty())
        }

        ty::TyArray(ety, _) | ty::TySlice(ety) => {
            // single-element containers, behave like their element
            dtorck_constraint_for_ty(tcx, span, for_ty, depth + 1, ety)
        }

        ty::TyTuple(tys) => tys.iter()
            .map(|ty| dtorck_constraint_for_ty(tcx, span, for_ty, depth + 1, ty))
            .collect(),

        ty::TyClosure(def_id, substs) => substs
            .upvar_tys(def_id, tcx)
            .map(|ty| dtorck_constraint_for_ty(tcx, span, for_ty, depth + 1, ty))
            .collect(),

        ty::TyGenerator(def_id, substs, _) => {
            // Note that the interior types are ignored here.
            // Any type reachable inside the interior must also be reachable
            // through the upvars.
            substs
                .upvar_tys(def_id, tcx)
                .map(|ty| dtorck_constraint_for_ty(tcx, span, for_ty, depth + 1, ty))
                .collect()
        }

        ty::TyAdt(def, substs) => {
            let DtorckConstraint {
                dtorck_types,
                outlives,
                overflows,
            } = tcx.at(span).adt_dtorck_constraint(def.did)?;
            Ok(DtorckConstraint {
                // FIXME: we can try to recursively `dtorck_constraint_on_ty`
                // there, but that needs some way to handle cycles.
                dtorck_types: dtorck_types.subst(tcx, substs),
                outlives: outlives.subst(tcx, substs),
                overflows: overflows.subst(tcx, substs),
            })
        }

        // Objects must be alive in order for their destructor
        // to be called.
        ty::TyDynamic(..) => Ok(DtorckConstraint {
            outlives: vec![ty.into()],
            dtorck_types: vec![],
            overflows: vec![],
        }),

        // Types that can't be resolved. Pass them forward.
        ty::TyProjection(..) | ty::TyAnon(..) | ty::TyParam(..) => Ok(DtorckConstraint {
            outlives: vec![],
            dtorck_types: vec![ty],
            overflows: vec![],
        }),

        ty::TyInfer(..) | ty::TyError => {
            // By the time this code runs, all type variables ought to
            // be fully resolved.
            Err(NoSolution)
        }
    };

    debug!("dtorck_constraint_for_ty({:?}) = {:?}", ty, result);
    result
}

/// Calculates the dtorck constraint for a type.
crate fn adt_dtorck_constraint<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    def_id: DefId,
) -> Result<DtorckConstraint<'tcx>, NoSolution> {
    let def = tcx.adt_def(def_id);
    let span = tcx.def_span(def_id);
    debug!("dtorck_constraint: {:?}", def);

    if def.is_phantom_data() {
        let result = DtorckConstraint {
            outlives: vec![],
            dtorck_types: vec![tcx.mk_param_from_def(&tcx.generics_of(def_id).types[0])],
            overflows: vec![],
        };
        debug!("dtorck_constraint: {:?} => {:?}", def, result);
        return Ok(result);
    }

    let mut result = def.all_fields()
        .map(|field| tcx.type_of(field.did))
        .map(|fty| dtorck_constraint_for_ty(tcx, span, fty, 0, fty))
        .collect::<Result<DtorckConstraint, NoSolution>>()?;
    result.outlives.extend(tcx.destructor_constraints(def));
    dedup_dtorck_constraint(&mut result);

    debug!("dtorck_constraint: {:?} => {:?}", def, result);

    Ok(result)
}

fn dedup_dtorck_constraint<'tcx>(c: &mut DtorckConstraint<'tcx>) {
    let mut outlives = FxHashSet();
    let mut dtorck_types = FxHashSet();

    c.outlives.retain(|&val| outlives.replace(val).is_none());
    c.dtorck_types
        .retain(|&val| dtorck_types.replace(val).is_none());
}
