use rustc_hir::{Movability, Mutability};
use rustc_infer::traits::query::NoSolution;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt};

use crate::solve::EvalCtxt;

// Calculates the constituent types of a type for `auto trait` purposes.
//
// For types with an "existential" binder, i.e. generator witnesses, we also
// instantiate the binder with placeholders eagerly.
pub(in crate::solve) fn instantiate_constituent_tys_for_auto_trait<'tcx>(
    ecx: &EvalCtxt<'_, 'tcx>,
    ty: Ty<'tcx>,
) -> Result<Vec<Ty<'tcx>>, NoSolution> {
    let tcx = ecx.tcx();
    match *ty.kind() {
        ty::Uint(_)
        | ty::Int(_)
        | ty::Bool
        | ty::Float(_)
        | ty::FnDef(..)
        | ty::FnPtr(_)
        | ty::Error(_)
        | ty::Never
        | ty::Char => Ok(vec![]),

        // Treat `str` like it's defined as `struct str([u8]);`
        ty::Str => Ok(vec![tcx.mk_slice(tcx.types.u8)]),

        ty::Dynamic(..)
        | ty::Param(..)
        | ty::Foreign(..)
        | ty::Alias(ty::Projection, ..)
        | ty::Placeholder(..)
        | ty::Bound(..)
        | ty::Infer(_) => {
            bug!("unexpected type `{ty}`")
        }

        ty::RawPtr(ty::TypeAndMut { ty: element_ty, .. }) | ty::Ref(_, element_ty, _) => {
            Ok(vec![element_ty])
        }

        ty::Array(element_ty, _) | ty::Slice(element_ty) => Ok(vec![element_ty]),

        ty::Tuple(ref tys) => {
            // (T1, ..., Tn) -- meets any bound that all of T1...Tn meet
            Ok(tys.iter().collect())
        }

        ty::Closure(_, ref substs) => Ok(vec![substs.as_closure().tupled_upvars_ty()]),

        ty::Generator(_, ref substs, _) => {
            let generator_substs = substs.as_generator();
            Ok(vec![generator_substs.tupled_upvars_ty(), generator_substs.witness()])
        }

        ty::GeneratorWitness(types) => Ok(ecx.instantiate_binder_with_placeholders(types).to_vec()),

        ty::GeneratorWitnessMIR(def_id, substs) => Ok(ecx
            .tcx()
            .generator_hidden_types(def_id)
            .map(|bty| {
                ecx.instantiate_binder_with_placeholders(replace_erased_lifetimes_with_bound_vars(
                    tcx,
                    bty.subst(tcx, substs),
                ))
            })
            .collect()),

        // For `PhantomData<T>`, we pass `T`.
        ty::Adt(def, substs) if def.is_phantom_data() => Ok(vec![substs.type_at(0)]),

        ty::Adt(def, substs) => Ok(def.all_fields().map(|f| f.ty(tcx, substs)).collect()),

        ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) => {
            // We can resolve the `impl Trait` to its concrete type,
            // which enforces a DAG between the functions requiring
            // the auto trait bounds in question.
            Ok(vec![tcx.type_of(def_id).subst(tcx, substs)])
        }
    }
}

pub(in crate::solve) fn replace_erased_lifetimes_with_bound_vars<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> ty::Binder<'tcx, Ty<'tcx>> {
    debug_assert!(!ty.has_late_bound_regions());
    let mut counter = 0;
    let ty = tcx.fold_regions(ty, |mut r, current_depth| {
        if let ty::ReErased = r.kind() {
            let br =
                ty::BoundRegion { var: ty::BoundVar::from_u32(counter), kind: ty::BrAnon(None) };
            counter += 1;
            r = tcx.mk_re_late_bound(current_depth, br);
        }
        r
    });
    let bound_vars = tcx.mk_bound_variable_kinds_from_iter(
        (0..counter).map(|_| ty::BoundVariableKind::Region(ty::BrAnon(None))),
    );
    ty::Binder::bind_with_vars(ty, bound_vars)
}

pub(in crate::solve) fn instantiate_constituent_tys_for_sized_trait<'tcx>(
    ecx: &EvalCtxt<'_, 'tcx>,
    ty: Ty<'tcx>,
) -> Result<Vec<Ty<'tcx>>, NoSolution> {
    match *ty.kind() {
        ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
        | ty::Uint(_)
        | ty::Int(_)
        | ty::Bool
        | ty::Float(_)
        | ty::FnDef(..)
        | ty::FnPtr(_)
        | ty::RawPtr(..)
        | ty::Char
        | ty::Ref(..)
        | ty::Generator(..)
        | ty::GeneratorWitness(..)
        | ty::GeneratorWitnessMIR(..)
        | ty::Array(..)
        | ty::Closure(..)
        | ty::Never
        | ty::Dynamic(_, _, ty::DynStar)
        | ty::Error(_) => Ok(vec![]),

        ty::Str
        | ty::Slice(_)
        | ty::Dynamic(..)
        | ty::Foreign(..)
        | ty::Alias(..)
        | ty::Param(_)
        | ty::Placeholder(..) => Err(NoSolution),

        ty::Bound(..)
        | ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
            bug!("unexpected type `{ty}`")
        }

        ty::Tuple(tys) => Ok(tys.to_vec()),

        ty::Adt(def, substs) => {
            let sized_crit = def.sized_constraint(ecx.tcx());
            Ok(sized_crit
                .0
                .iter()
                .map(|ty| sized_crit.rebind(*ty).subst(ecx.tcx(), substs))
                .collect())
        }
    }
}

pub(in crate::solve) fn instantiate_constituent_tys_for_copy_clone_trait<'tcx>(
    ecx: &EvalCtxt<'_, 'tcx>,
    ty: Ty<'tcx>,
) -> Result<Vec<Ty<'tcx>>, NoSolution> {
    match *ty.kind() {
        ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
        | ty::FnDef(..)
        | ty::FnPtr(_)
        | ty::Error(_) => Ok(vec![]),

        // Implementations are provided in core
        ty::Uint(_)
        | ty::Int(_)
        | ty::Bool
        | ty::Float(_)
        | ty::Char
        | ty::RawPtr(..)
        | ty::Never
        | ty::Ref(_, _, Mutability::Not)
        | ty::Array(..) => Err(NoSolution),

        ty::Dynamic(..)
        | ty::Str
        | ty::Slice(_)
        | ty::Generator(_, _, Movability::Static)
        | ty::Foreign(..)
        | ty::Ref(_, _, Mutability::Mut)
        | ty::Adt(_, _)
        | ty::Alias(_, _)
        | ty::Param(_)
        | ty::Placeholder(..) => Err(NoSolution),

        ty::Bound(..)
        | ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
            bug!("unexpected type `{ty}`")
        }

        ty::Tuple(tys) => Ok(tys.to_vec()),

        ty::Closure(_, substs) => Ok(vec![substs.as_closure().tupled_upvars_ty()]),

        ty::Generator(_, substs, Movability::Movable) => {
            if ecx.tcx().features().generator_clone {
                let generator = substs.as_generator();
                Ok(vec![generator.tupled_upvars_ty(), generator.witness()])
            } else {
                Err(NoSolution)
            }
        }

        ty::GeneratorWitness(types) => Ok(ecx.instantiate_binder_with_placeholders(types).to_vec()),

        ty::GeneratorWitnessMIR(def_id, substs) => Ok(ecx
            .tcx()
            .generator_hidden_types(def_id)
            .map(|bty| {
                ecx.instantiate_binder_with_placeholders(replace_erased_lifetimes_with_bound_vars(
                    ecx.tcx(),
                    bty.subst(ecx.tcx(), substs),
                ))
            })
            .collect()),
    }
}

// Returns a binder of the tupled inputs types and output type from a builtin callable type.
pub(in crate::solve) fn extract_tupled_inputs_and_output_from_callable<'tcx>(
    tcx: TyCtxt<'tcx>,
    self_ty: Ty<'tcx>,
    goal_kind: ty::ClosureKind,
) -> Result<Option<ty::Binder<'tcx, (Ty<'tcx>, Ty<'tcx>)>>, NoSolution> {
    match *self_ty.kind() {
        // keep this in sync with assemble_fn_pointer_candidates until the old solver is removed.
        ty::FnDef(def_id, substs) => {
            let sig = tcx.fn_sig(def_id);
            if sig.skip_binder().is_fn_trait_compatible()
                && tcx.codegen_fn_attrs(def_id).target_features.is_empty()
            {
                Ok(Some(
                    sig.subst(tcx, substs)
                        .map_bound(|sig| (tcx.mk_tup(sig.inputs()), sig.output())),
                ))
            } else {
                Err(NoSolution)
            }
        }
        // keep this in sync with assemble_fn_pointer_candidates until the old solver is removed.
        ty::FnPtr(sig) => {
            if sig.is_fn_trait_compatible() {
                Ok(Some(sig.map_bound(|sig| (tcx.mk_tup(sig.inputs()), sig.output()))))
            } else {
                Err(NoSolution)
            }
        }
        ty::Closure(_, substs) => {
            let closure_substs = substs.as_closure();
            match closure_substs.kind_ty().to_opt_closure_kind() {
                // If the closure's kind doesn't extend the goal kind,
                // then the closure doesn't implement the trait.
                Some(closure_kind) => {
                    if !closure_kind.extends(goal_kind) {
                        return Err(NoSolution);
                    }
                }
                // Closure kind is not yet determined, so we return ambiguity unless
                // the expected kind is `FnOnce` as that is always implemented.
                None => {
                    if goal_kind != ty::ClosureKind::FnOnce {
                        return Ok(None);
                    }
                }
            }
            Ok(Some(closure_substs.sig().map_bound(|sig| (sig.inputs()[0], sig.output()))))
        }
        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Adt(_, _)
        | ty::Foreign(_)
        | ty::Str
        | ty::Array(_, _)
        | ty::Slice(_)
        | ty::RawPtr(_)
        | ty::Ref(_, _, _)
        | ty::Dynamic(_, _, _)
        | ty::Generator(_, _, _)
        | ty::GeneratorWitness(_)
        | ty::GeneratorWitnessMIR(..)
        | ty::Never
        | ty::Tuple(_)
        | ty::Alias(_, _)
        | ty::Param(_)
        | ty::Placeholder(..)
        | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
        | ty::Error(_) => Err(NoSolution),

        ty::Bound(..)
        | ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
            bug!("unexpected type `{self_ty}`")
        }
    }
}

/// Assemble a list of predicates that need to hold for a trait implementation
/// to be WF.
pub(in crate::solve) fn requirements_for_trait_wf<'tcx>(
    ecx: &EvalCtxt<'_, 'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
) -> Vec<ty::Predicate<'tcx>> {
    let tcx = ecx.tcx();
    let mut requirements = vec![];
    requirements.extend(
        tcx.super_predicates_of(trait_ref.def_id).instantiate(tcx, trait_ref.substs).predicates,
    );
    for item in tcx.associated_items(trait_ref.def_id).in_definition_order() {
        // FIXME(associated_const_equality): Also add associated consts to
        // the requirements here.
        if item.kind == ty::AssocKind::Type {
            requirements.extend(tcx.item_bounds(item.def_id).subst(tcx, trait_ref.substs));
        }
    }
    requirements
}
