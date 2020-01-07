//! Miscellaneous type-system utilities that are too small to deserve their own modules.

use crate::middle::lang_items;
use crate::traits::{self, ObligationCause};
use crate::ty::util::NeedsDrop;
use crate::ty::{self, Ty, TyCtxt, TypeFoldable};

use rustc_hir as hir;
use rustc_span::DUMMY_SP;

#[derive(Clone)]
pub enum CopyImplementationError<'tcx> {
    InfrigingFields(Vec<&'tcx ty::FieldDef>),
    NotAnAdt,
    HasDestructor,
}

pub fn can_type_implement_copy(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    self_type: Ty<'tcx>,
) -> Result<(), CopyImplementationError<'tcx>> {
    // FIXME: (@jroesch) float this code up
    tcx.infer_ctxt().enter(|infcx| {
        let (adt, substs) = match self_type.kind {
            // These types used to have a builtin impl.
            // Now libcore provides that impl.
            ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::Char
            | ty::RawPtr(..)
            | ty::Never
            | ty::Ref(_, _, hir::Mutability::Not) => return Ok(()),

            ty::Adt(adt, substs) => (adt, substs),

            _ => return Err(CopyImplementationError::NotAnAdt),
        };

        let mut infringing = Vec::new();
        for variant in &adt.variants {
            for field in &variant.fields {
                let ty = field.ty(tcx, substs);
                if ty.references_error() {
                    continue;
                }
                let span = tcx.def_span(field.did);
                let cause = ObligationCause { span, ..ObligationCause::dummy() };
                let ctx = traits::FulfillmentContext::new();
                match traits::fully_normalize(&infcx, ctx, cause, param_env, &ty) {
                    Ok(ty) => {
                        if !infcx.type_is_copy_modulo_regions(param_env, ty, span) {
                            infringing.push(field);
                        }
                    }
                    Err(errors) => {
                        infcx.report_fulfillment_errors(&errors, None, false);
                    }
                };
            }
        }
        if !infringing.is_empty() {
            return Err(CopyImplementationError::InfrigingFields(infringing));
        }
        if adt.has_dtor(tcx) {
            return Err(CopyImplementationError::HasDestructor);
        }

        Ok(())
    })
}

fn is_copy_raw<'tcx>(tcx: TyCtxt<'tcx>, query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
    is_item_raw(tcx, query, lang_items::CopyTraitLangItem)
}

fn is_sized_raw<'tcx>(tcx: TyCtxt<'tcx>, query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
    is_item_raw(tcx, query, lang_items::SizedTraitLangItem)
}

fn is_freeze_raw<'tcx>(tcx: TyCtxt<'tcx>, query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
    is_item_raw(tcx, query, lang_items::FreezeTraitLangItem)
}

fn is_item_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
    item: lang_items::LangItem,
) -> bool {
    let (param_env, ty) = query.into_parts();
    let trait_def_id = tcx.require_lang_item(item, None);
    tcx.infer_ctxt().enter(|infcx| {
        traits::type_known_to_meet_bound_modulo_regions(
            &infcx,
            param_env,
            ty,
            trait_def_id,
            DUMMY_SP,
        )
    })
}

fn needs_drop_raw<'tcx>(tcx: TyCtxt<'tcx>, query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> NeedsDrop {
    let (param_env, ty) = query.into_parts();

    let needs_drop = |ty: Ty<'tcx>| -> bool { tcx.needs_drop_raw(param_env.and(ty)).0 };

    assert!(!ty.needs_infer());

    NeedsDrop(match ty.kind {
        // Fast-path for primitive types
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
        | ty::Str => false,

        // Foreign types can never have destructors
        ty::Foreign(..) => false,

        // `ManuallyDrop` doesn't have a destructor regardless of field types.
        ty::Adt(def, _) if Some(def.did) == tcx.lang_items().manually_drop() => false,

        // Issue #22536: We first query `is_copy_modulo_regions`.  It sees a
        // normalized version of the type, and therefore will definitely
        // know whether the type implements Copy (and thus needs no
        // cleanup/drop/zeroing) ...
        _ if ty.is_copy_modulo_regions(tcx, param_env, DUMMY_SP) => false,

        // ... (issue #22536 continued) but as an optimization, still use
        // prior logic of asking for the structural "may drop".

        // FIXME(#22815): Note that this is a conservative heuristic;
        // it may report that the type "may drop" when actual type does
        // not actually have a destructor associated with it. But since
        // the type absolutely did not have the `Copy` bound attached
        // (see above), it is sound to treat it as having a destructor.

        // User destructors are the only way to have concrete drop types.
        ty::Adt(def, _) if def.has_dtor(tcx) => true,

        // Can refer to a type which may drop.
        // FIXME(eddyb) check this against a ParamEnv.
        ty::Dynamic(..)
        | ty::Projection(..)
        | ty::Param(_)
        | ty::Bound(..)
        | ty::Placeholder(..)
        | ty::Opaque(..)
        | ty::Infer(_)
        | ty::Error => true,

        ty::UnnormalizedProjection(..) => bug!("only used with chalk-engine"),

        // Zero-length arrays never contain anything to drop.
        ty::Array(_, len) if len.try_eval_usize(tcx, param_env) == Some(0) => false,

        // Structural recursion.
        ty::Array(ty, _) | ty::Slice(ty) => needs_drop(ty),

        ty::Closure(def_id, ref substs) => {
            substs.as_closure().upvar_tys(def_id, tcx).any(needs_drop)
        }

        // Pessimistically assume that all generators will require destructors
        // as we don't know if a destructor is a noop or not until after the MIR
        // state transformation pass
        ty::Generator(..) => true,

        ty::Tuple(..) => ty.tuple_fields().any(needs_drop),

        // unions don't have destructors because of the child types,
        // only if they manually implement `Drop` (handled above).
        ty::Adt(def, _) if def.is_union() => false,

        ty::Adt(def, substs) => def
            .variants
            .iter()
            .any(|variant| variant.fields.iter().any(|field| needs_drop(field.ty(tcx, substs)))),
    })
}

pub fn provide(providers: &mut ty::query::Providers<'_>) {
    *providers = ty::query::Providers {
        is_copy_raw,
        is_sized_raw,
        is_freeze_raw,
        needs_drop_raw,
        ..*providers
    };
}
