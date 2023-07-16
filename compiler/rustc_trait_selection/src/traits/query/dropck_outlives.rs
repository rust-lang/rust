use crate::traits::query::normalize::QueryNormalizeExt;
use crate::traits::query::NoSolution;
use crate::traits::{Normalized, ObligationCause, ObligationCtxt};

use rustc_data_structures::fx::FxHashSet;
use rustc_middle::traits::query::{DropckConstraint, DropckOutlivesResult};
use rustc_middle::ty::{self, EarlyBinder, ParamEnvAnd, Ty, TyCtxt};
use rustc_span::source_map::{Span, DUMMY_SP};

/// This returns true if the type `ty` is "trivial" for
/// dropck-outlives -- that is, if it doesn't require any types to
/// outlive. This is similar but not *quite* the same as the
/// `needs_drop` test in the compiler already -- that is, for every
/// type T for which this function return true, needs-drop would
/// return `false`. But the reverse does not hold: in particular,
/// `needs_drop` returns false for `PhantomData`, but it is not
/// trivial for dropck-outlives.
///
/// Note also that `needs_drop` requires a "global" type (i.e., one
/// with erased regions), but this function does not.
///
// FIXME(@lcnr): remove this module and move this function somewhere else.
pub fn trivial_dropck_outlives<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind() {
        // None of these types have a destructor and hence they do not
        // require anything in particular to outlive the dtor's
        // execution.
        ty::Infer(ty::FreshIntTy(_))
        | ty::Infer(ty::FreshFloatTy(_))
        | ty::Bool
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Never
        | ty::FnDef(..)
        | ty::FnPtr(_)
        | ty::Char
        | ty::GeneratorWitness(..)
        | ty::GeneratorWitnessMIR(..)
        | ty::RawPtr(_)
        | ty::Ref(..)
        | ty::Str
        | ty::Foreign(..)
        | ty::Error(_) => true,

        // [T; N] and [T] have same properties as T.
        ty::Array(ty, _) | ty::Slice(ty) => trivial_dropck_outlives(tcx, *ty),

        // (T1..Tn) and closures have same properties as T1..Tn --
        // check if *all* of them are trivial.
        ty::Tuple(tys) => tys.iter().all(|t| trivial_dropck_outlives(tcx, t)),
        ty::Closure(_, ref args) => {
            trivial_dropck_outlives(tcx, args.as_closure().tupled_upvars_ty())
        }

        ty::Adt(def, _) => {
            if Some(def.did()) == tcx.lang_items().manually_drop() {
                // `ManuallyDrop` never has a dtor.
                true
            } else {
                // Other types might. Moreover, PhantomData doesn't
                // have a dtor, but it is considered to own its
                // content, so it is non-trivial. Unions can have `impl Drop`,
                // and hence are non-trivial as well.
                false
            }
        }

        // The following *might* require a destructor: needs deeper inspection.
        ty::Dynamic(..)
        | ty::Alias(..)
        | ty::Param(_)
        | ty::Placeholder(..)
        | ty::Infer(_)
        | ty::Bound(..)
        | ty::Generator(..) => false,
    }
}

pub fn compute_dropck_outlives_inner<'tcx>(
    ocx: &ObligationCtxt<'_, 'tcx>,
    goal: ParamEnvAnd<'tcx, Ty<'tcx>>,
) -> Result<DropckOutlivesResult<'tcx>, NoSolution> {
    let tcx = ocx.infcx.tcx;
    let ParamEnvAnd { param_env, value: for_ty } = goal;

    let mut result = DropckOutlivesResult { kinds: vec![], overflows: vec![] };

    // A stack of types left to process. Each round, we pop
    // something from the stack and invoke
    // `dtorck_constraint_for_ty_inner`. This may produce new types that
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
    // invoke `dtorck_constraint_for_ty_inner` which will expand `A`
    // into the types of its fields `(B, Vec<A>)`. These will get
    // pushed onto the stack. Eventually, expanding `Vec<A>` will
    // lead to us trying to push `A` a second time -- to prevent
    // infinite recursion, we notice that `A` was already pushed
    // once and stop.
    let mut ty_stack = vec![(for_ty, 0)];

    // Set used to detect infinite recursion.
    let mut ty_set = FxHashSet::default();

    let cause = ObligationCause::dummy();
    let mut constraints = DropckConstraint::empty();
    while let Some((ty, depth)) = ty_stack.pop() {
        debug!(
            "{} kinds, {} overflows, {} ty_stack",
            result.kinds.len(),
            result.overflows.len(),
            ty_stack.len()
        );
        dtorck_constraint_for_ty_inner(tcx, DUMMY_SP, for_ty, depth, ty, &mut constraints)?;

        // "outlives" represent types/regions that may be touched
        // by a destructor.
        result.kinds.append(&mut constraints.outlives);
        result.overflows.append(&mut constraints.overflows);

        // If we have even one overflow, we should stop trying to evaluate further --
        // chances are, the subsequent overflows for this evaluation won't provide useful
        // information and will just decrease the speed at which we can emit these errors
        // (since we'll be printing for just that much longer for the often enormous types
        // that result here).
        if !result.overflows.is_empty() {
            break;
        }

        // dtorck types are "types that will get dropped but which
        // do not themselves define a destructor", more or less. We have
        // to push them onto the stack to be expanded.
        for ty in constraints.dtorck_types.drain(..) {
            let Normalized { value: ty, obligations } =
                ocx.infcx.at(&cause, param_env).query_normalize(ty)?;
            ocx.register_obligations(obligations);

            debug!("dropck_outlives: ty from dtorck_types = {:?}", ty);

            match ty.kind() {
                // All parameters live for the duration of the
                // function.
                ty::Param(..) => {}

                // A projection that we couldn't resolve - it
                // might have a destructor.
                ty::Alias(..) => {
                    result.kinds.push(ty.into());
                }

                _ => {
                    if ty_set.insert(ty) {
                        ty_stack.push((ty, depth + 1));
                    }
                }
            }
        }
    }

    debug!("dropck_outlives: result = {:#?}", result);
    Ok(result)
}

/// Returns a set of constraints that needs to be satisfied in
/// order for `ty` to be valid for destruction.
pub fn dtorck_constraint_for_ty_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    for_ty: Ty<'tcx>,
    depth: usize,
    ty: Ty<'tcx>,
    constraints: &mut DropckConstraint<'tcx>,
) -> Result<(), NoSolution> {
    debug!("dtorck_constraint_for_ty_inner({:?}, {:?}, {:?}, {:?})", span, for_ty, depth, ty);

    if !tcx.recursion_limit().value_within_limit(depth) {
        constraints.overflows.push(ty);
        return Ok(());
    }

    if trivial_dropck_outlives(tcx, ty) {
        return Ok(());
    }

    match ty.kind() {
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
        | ty::GeneratorWitness(..)
        | ty::GeneratorWitnessMIR(..) => {
            // these types never have a destructor
        }

        ty::Array(ety, _) | ty::Slice(ety) => {
            // single-element containers, behave like their element
            rustc_data_structures::stack::ensure_sufficient_stack(|| {
                dtorck_constraint_for_ty_inner(tcx, span, for_ty, depth + 1, *ety, constraints)
            })?;
        }

        ty::Tuple(tys) => rustc_data_structures::stack::ensure_sufficient_stack(|| {
            for ty in tys.iter() {
                dtorck_constraint_for_ty_inner(tcx, span, for_ty, depth + 1, ty, constraints)?;
            }
            Ok::<_, NoSolution>(())
        })?,

        ty::Closure(_, args) => {
            if !args.as_closure().is_valid() {
                // By the time this code runs, all type variables ought to
                // be fully resolved.

                tcx.sess.delay_span_bug(
                    span,
                    format!("upvar_tys for closure not found. Expected capture information for closure {ty}",),
                );
                return Err(NoSolution);
            }

            rustc_data_structures::stack::ensure_sufficient_stack(|| {
                for ty in args.as_closure().upvar_tys() {
                    dtorck_constraint_for_ty_inner(tcx, span, for_ty, depth + 1, ty, constraints)?;
                }
                Ok::<_, NoSolution>(())
            })?
        }

        ty::Generator(_, args, _movability) => {
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
            // derived from lifetimes attached to the upvars and resume
            // argument, and we *do* incorporate those here.

            if !args.as_generator().is_valid() {
                // By the time this code runs, all type variables ought to
                // be fully resolved.
                tcx.sess.delay_span_bug(
                    span,
                    format!("upvar_tys for generator not found. Expected capture information for generator {ty}",),
                );
                return Err(NoSolution);
            }

            constraints.outlives.extend(
                args.as_generator().upvar_tys().map(|t| -> ty::GenericArg<'tcx> { t.into() }),
            );
            constraints.outlives.push(args.as_generator().resume_ty().into());
        }

        ty::Adt(def, args) => {
            let DropckConstraint { dtorck_types, outlives, overflows } =
                tcx.at(span).adt_dtorck_constraint(def.did())?;
            // FIXME: we can try to recursively `dtorck_constraint_on_ty`
            // there, but that needs some way to handle cycles.
            constraints
                .dtorck_types
                .extend(dtorck_types.iter().map(|t| EarlyBinder::bind(*t).instantiate(tcx, args)));
            constraints
                .outlives
                .extend(outlives.iter().map(|t| EarlyBinder::bind(*t).instantiate(tcx, args)));
            constraints
                .overflows
                .extend(overflows.iter().map(|t| EarlyBinder::bind(*t).instantiate(tcx, args)));
        }

        // Objects must be alive in order for their destructor
        // to be called.
        ty::Dynamic(..) => {
            constraints.outlives.push(ty.into());
        }

        // Types that can't be resolved. Pass them forward.
        ty::Alias(..) | ty::Param(..) => {
            constraints.dtorck_types.push(ty);
        }

        ty::Placeholder(..) | ty::Bound(..) | ty::Infer(..) | ty::Error(_) => {
            // By the time this code runs, all type variables ought to
            // be fully resolved.
            return Err(NoSolution);
        }
    }

    Ok(())
}
