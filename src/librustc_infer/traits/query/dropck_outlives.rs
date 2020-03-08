use crate::infer::at::At;
use crate::infer::canonical::OriginalQueryValues;
use crate::infer::InferOk;

use rustc::ty::subst::GenericArg;
use rustc::ty::{self, Ty, TyCtxt};

pub use rustc::traits::query::{DropckOutlivesResult, DtorckConstraint};

impl<'cx, 'tcx> At<'cx, 'tcx> {
    /// Given a type `ty` of some value being dropped, computes a set
    /// of "kinds" (types, regions) that must be outlive the execution
    /// of the destructor. These basically correspond to data that the
    /// destructor might access. This is used during regionck to
    /// impose "outlives" constraints on any lifetimes referenced
    /// within.
    ///
    /// The rules here are given by the "dropck" RFCs, notably [#1238]
    /// and [#1327]. This is a fixed-point computation, where we
    /// explore all the data that will be dropped (transitively) when
    /// a value of type `ty` is dropped. For each type T that will be
    /// dropped and which has a destructor, we must assume that all
    /// the types/regions of T are live during the destructor, unless
    /// they are marked with a special attribute (`#[may_dangle]`).
    ///
    /// [#1238]: https://github.com/rust-lang/rfcs/blob/master/text/1238-nonparametric-dropck.md
    /// [#1327]: https://github.com/rust-lang/rfcs/blob/master/text/1327-dropck-param-eyepatch.md
    pub fn dropck_outlives(&self, ty: Ty<'tcx>) -> InferOk<'tcx, Vec<GenericArg<'tcx>>> {
        debug!("dropck_outlives(ty={:?}, param_env={:?})", ty, self.param_env,);

        // Quick check: there are a number of cases that we know do not require
        // any destructor.
        let tcx = self.infcx.tcx;
        if trivial_dropck_outlives(tcx, ty) {
            return InferOk { value: vec![], obligations: vec![] };
        }

        let mut orig_values = OriginalQueryValues::default();
        let c_ty = self.infcx.canonicalize_query(&self.param_env.and(ty), &mut orig_values);
        let span = self.cause.span;
        debug!("c_ty = {:?}", c_ty);
        if let Ok(result) = &tcx.dropck_outlives(c_ty) {
            if result.is_proven() {
                if let Ok(InferOk { value, obligations }) =
                    self.infcx.instantiate_query_response_and_region_obligations(
                        self.cause,
                        self.param_env,
                        &orig_values,
                        result,
                    )
                {
                    let ty = self.infcx.resolve_vars_if_possible(&ty);
                    let kinds = value.into_kinds_reporting_overflows(tcx, span, ty);
                    return InferOk { value: kinds, obligations };
                }
            }
        }

        // Errors and ambiuity in dropck occur in two cases:
        // - unresolved inference variables at the end of typeck
        // - non well-formed types where projections cannot be resolved
        // Either of these should have created an error before.
        tcx.sess.delay_span_bug(span, "dtorck encountered internal error");

        InferOk { value: vec![], obligations: vec![] }
    }
}

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
pub fn trivial_dropck_outlives<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind {
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
        | ty::RawPtr(_)
        | ty::Ref(..)
        | ty::Str
        | ty::Foreign(..)
        | ty::Error => true,

        // [T; N] and [T] have same properties as T.
        ty::Array(ty, _) | ty::Slice(ty) => trivial_dropck_outlives(tcx, ty),

        // (T1..Tn) and closures have same properties as T1..Tn --
        // check if *any* of those are trivial.
        ty::Tuple(ref tys) => tys.iter().all(|t| trivial_dropck_outlives(tcx, t.expect_ty())),
        ty::Closure(def_id, ref substs) => {
            substs.as_closure().upvar_tys(def_id, tcx).all(|t| trivial_dropck_outlives(tcx, t))
        }

        ty::Adt(def, _) => {
            if Some(def.did) == tcx.lang_items().manually_drop() {
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
        | ty::Projection(..)
        | ty::Param(_)
        | ty::Opaque(..)
        | ty::Placeholder(..)
        | ty::Infer(_)
        | ty::Bound(..)
        | ty::Generator(..) => false,

        ty::UnnormalizedProjection(..) => bug!("only used with chalk-engine"),
    }
}
