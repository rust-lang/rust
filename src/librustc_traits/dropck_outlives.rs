use rustc::hir::def_id::DefId;
use rustc::infer::canonical::{Canonical, QueryResponse};
use rustc::traits::query::dropck_outlives::{DropckOutlivesResult, DtorckConstraint};
use rustc::traits::query::{CanonicalTyGoal, NoSolution};
use rustc::traits::{TraitEngine, Normalized, ObligationCause, TraitEngineExt};
use rustc::ty::query::Providers;
use rustc::ty::subst::{Subst, InternalSubsts};
use rustc::ty::{self, ParamEnvAnd, Ty, TyCtxt};
use rustc::util::nodemap::FxHashSet;
use syntax::source_map::{Span, DUMMY_SP};

crate fn provide(p: &mut Providers<'_>) {
    *p = Providers {
        dropck_outlives,
        adt_dtorck_constraint,
        ..*p
    };
}

fn dropck_outlives<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonical_goal: CanonicalTyGoal<'tcx>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, DropckOutlivesResult<'tcx>>>, NoSolution> {
    debug!("dropck_outlives(goal={:#?})", canonical_goal);

    tcx.infer_ctxt().enter_with_canonical(
        DUMMY_SP,
        &canonical_goal,
        |ref infcx, goal, canonical_inference_vars| {
            let tcx = infcx.tcx;
            let ParamEnvAnd {
                param_env,
                value: for_ty,
            } = goal;

            let mut result = DropckOutlivesResult {
                kinds: vec![],
                overflows: vec![],
            };

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
            // infinite recursion, we notice that `A` was already pushed
            // once and stop.
            let mut ty_stack = vec![(for_ty, 0)];

            // Set used to detect infinite recursion.
            let mut ty_set = FxHashSet::default();

            let mut fulfill_cx = TraitEngine::new(infcx.tcx);

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
                                ty::Param(..) => {}

                                // A projection that we couldn't resolve - it
                                // might have a destructor.
                                ty::Projection(..) | ty::Opaque(..) => {
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

            infcx.make_canonicalized_query_response(
                canonical_inference_vars,
                result,
                &mut *fulfill_cx
            )
        },
    )
}

/// Returns a set of constraints that needs to be satisfied in
/// order for `ty` to be valid for destruction.
fn dtorck_constraint_for_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
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
        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Str
        | ty::Never
        | ty::Foreign(..)
        | ty::RawPtr(..)
        | ty::Ref(..)
        | ty::FnDef(..)
        | ty::FnPtr(_)
        | ty::GeneratorWitness(..) => {
            // these types never have a destructor
            Ok(DtorckConstraint::empty())
        }

        ty::Array(ety, _) | ty::Slice(ety) => {
            // single-element containers, behave like their element
            dtorck_constraint_for_ty(tcx, span, for_ty, depth + 1, ety)
        }

        ty::Tuple(tys) => tys.iter()
            .map(|ty| dtorck_constraint_for_ty(tcx, span, for_ty, depth + 1, ty.expect_ty()))
            .collect(),

        ty::Closure(def_id, substs) => substs
            .upvar_tys(def_id, tcx)
            .map(|ty| dtorck_constraint_for_ty(tcx, span, for_ty, depth + 1, ty))
            .collect(),

        ty::Generator(def_id, substs, _movability) => {
            // rust-lang/rust#49918: types can be constructed, stored
            // in the interior, and sit idle when generator yields
            // (and is subsequently dropped).
            //
            // It would be nice to descend into interior of a
            // generator to determine what effects dropping it might
            // have (by looking at any drop effects associated with
            // its interior).
            //
            // However, the interior's representation uses things like
            // GeneratorWitness that explicitly assume they are not
            // traversed in such a manner. So instead, we will
            // simplify things for now by treating all generators as
            // if they were like trait objects, where its upvars must
            // all be alive for the generator's (potential)
            // destructor.
            //
            // In particular, skipping over `_interior` is safe
            // because any side-effects from dropping `_interior` can
            // only take place through references with lifetimes
            // derived from lifetimes attached to the upvars, and we
            // *do* incorporate the upvars here.

            let constraint = DtorckConstraint {
                outlives: substs.upvar_tys(def_id, tcx).map(|t| t.into()).collect(),
                dtorck_types: vec![],
                overflows: vec![],
            };
            debug!(
                "dtorck_constraint: generator {:?} => {:?}",
                def_id, constraint
            );

            Ok(constraint)
        }

        ty::Adt(def, substs) => {
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
        ty::Dynamic(..) => Ok(DtorckConstraint {
            outlives: vec![ty.into()],
            dtorck_types: vec![],
            overflows: vec![],
        }),

        // Types that can't be resolved. Pass them forward.
        ty::Projection(..) | ty::Opaque(..) | ty::Param(..) => Ok(DtorckConstraint {
            outlives: vec![],
            dtorck_types: vec![ty],
            overflows: vec![],
        }),

        ty::UnnormalizedProjection(..) => bug!("only used with chalk-engine"),

        ty::Placeholder(..) | ty::Bound(..) | ty::Infer(..) | ty::Error => {
            // By the time this code runs, all type variables ought to
            // be fully resolved.
            Err(NoSolution)
        }
    };

    debug!("dtorck_constraint_for_ty({:?}) = {:?}", ty, result);
    result
}

/// Calculates the dtorck constraint for a type.
crate fn adt_dtorck_constraint(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> Result<DtorckConstraint<'_>, NoSolution> {
    let def = tcx.adt_def(def_id);
    let span = tcx.def_span(def_id);
    debug!("dtorck_constraint: {:?}", def);

    if def.is_phantom_data() {
        // The first generic parameter here is guaranteed to be a type because it's
        // `PhantomData`.
        let substs = InternalSubsts::identity_for_item(tcx, def_id);
        assert_eq!(substs.len(), 1);
        let result = DtorckConstraint {
            outlives: vec![],
            dtorck_types: vec![substs.type_at(0)],
            overflows: vec![],
        };
        debug!("dtorck_constraint: {:?} => {:?}", def, result);
        return Ok(result);
    }

    let mut result = def.all_fields()
        .map(|field| tcx.type_of(field.did))
        .map(|fty| dtorck_constraint_for_ty(tcx, span, fty, 0, fty))
        .collect::<Result<DtorckConstraint<'_>, NoSolution>>()?;
    result.outlives.extend(tcx.destructor_constraints(def));
    dedup_dtorck_constraint(&mut result);

    debug!("dtorck_constraint: {:?} => {:?}", def, result);

    Ok(result)
}

fn dedup_dtorck_constraint(c: &mut DtorckConstraint<'_>) {
    let mut outlives = FxHashSet::default();
    let mut dtorck_types = FxHashSet::default();

    c.outlives.retain(|&val| outlives.replace(val).is_none());
    c.dtorck_types
        .retain(|&val| dtorck_types.replace(val).is_none());
}
