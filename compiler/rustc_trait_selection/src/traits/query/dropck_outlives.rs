use rustc_middle::ty::{self, Ty, TyCtxt};

pub use rustc_middle::traits::query::{DropckConstraint, DropckOutlivesResult};

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
        ty::Closure(_, ref substs) => {
            trivial_dropck_outlives(tcx, substs.as_closure().tupled_upvars_ty())
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
