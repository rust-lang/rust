use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{def_id::DefId, Movability, Mutability};
use rustc_infer::traits::query::NoSolution;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
};

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
        ty::Str => Ok(vec![Ty::new_slice(tcx, tcx.types.u8)]),

        ty::Dynamic(..)
        | ty::Param(..)
        | ty::Foreign(..)
        | ty::Alias(ty::Projection | ty::Inherent | ty::Weak, ..)
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
    let ty = tcx.fold_regions(ty, |r, current_depth| match r.kind() {
        ty::ReErased => {
            let br =
                ty::BoundRegion { var: ty::BoundVar::from_u32(counter), kind: ty::BrAnon(None) };
            counter += 1;
            ty::Region::new_late_bound(tcx, current_depth, br)
        }
        // All free regions should be erased here.
        r => bug!("unexpected region: {r:?}"),
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
            Ok(sized_crit.subst_iter_copied(ecx.tcx(), substs).collect())
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
                        .map_bound(|sig| (Ty::new_tup(tcx, sig.inputs()), sig.output())),
                ))
            } else {
                Err(NoSolution)
            }
        }
        // keep this in sync with assemble_fn_pointer_candidates until the old solver is removed.
        ty::FnPtr(sig) => {
            if sig.is_fn_trait_compatible() {
                Ok(Some(sig.map_bound(|sig| (Ty::new_tup(tcx, sig.inputs()), sig.output()))))
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

/// Assemble a list of predicates that would be present on a theoretical
/// user impl for an object type. These predicates must be checked any time
/// we assemble a built-in object candidate for an object type, since they
/// are not implied by the well-formedness of the type.
///
/// For example, given the following traits:
///
/// ```rust,ignore (theoretical code)
/// trait Foo: Baz {
///     type Bar: Copy;
/// }
///
/// trait Baz {}
/// ```
///
/// For the dyn type `dyn Foo<Item = Ty>`, we can imagine there being a
/// pair of theoretical impls:
///
/// ```rust,ignore (theoretical code)
/// impl Foo for dyn Foo<Item = Ty>
/// where
///     Self: Baz,
///     <Self as Foo>::Bar: Copy,
/// {
///     type Bar = Ty;
/// }
///
/// impl Baz for dyn Foo<Item = Ty> {}
/// ```
///
/// However, in order to make such impls well-formed, we need to do an
/// additional step of eagerly folding the associated types in the where
/// clauses of the impl. In this example, that means replacing
/// `<Self as Foo>::Bar` with `Ty` in the first impl.
///
// FIXME: This is only necessary as `<Self as Trait>::Assoc: ItemBound`
// bounds in impls are trivially proven using the item bound candidates.
// This is unsound in general and once that is fixed, we don't need to
// normalize eagerly here. See https://github.com/lcnr/solver-woes/issues/9
// for more details.
pub(in crate::solve) fn predicates_for_object_candidate<'tcx>(
    ecx: &EvalCtxt<'_, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    object_bound: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
) -> Vec<ty::Clause<'tcx>> {
    let tcx = ecx.tcx();
    let mut requirements = vec![];
    requirements.extend(
        tcx.super_predicates_of(trait_ref.def_id).instantiate(tcx, trait_ref.substs).predicates,
    );
    for item in tcx.associated_items(trait_ref.def_id).in_definition_order() {
        // FIXME(associated_const_equality): Also add associated consts to
        // the requirements here.
        if item.kind == ty::AssocKind::Type {
            requirements.extend(tcx.item_bounds(item.def_id).subst_iter(tcx, trait_ref.substs));
        }
    }

    let mut replace_projection_with = FxHashMap::default();
    for bound in object_bound {
        if let ty::ExistentialPredicate::Projection(proj) = bound.skip_binder() {
            let proj = proj.with_self_ty(tcx, trait_ref.self_ty());
            let old_ty = replace_projection_with.insert(proj.def_id(), bound.rebind(proj));
            assert_eq!(
                old_ty,
                None,
                "{} has two substitutions: {} and {}",
                proj.projection_ty,
                proj.term,
                old_ty.unwrap()
            );
        }
    }

    requirements.fold_with(&mut ReplaceProjectionWith {
        ecx,
        param_env,
        mapping: replace_projection_with,
    })
}

struct ReplaceProjectionWith<'a, 'tcx> {
    ecx: &'a EvalCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    mapping: FxHashMap<DefId, ty::PolyProjectionPredicate<'tcx>>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ReplaceProjectionWith<'_, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.ecx.tcx()
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Alias(ty::Projection, alias_ty) = *ty.kind()
            && let Some(replacement) = self.mapping.get(&alias_ty.def_id)
        {
            // We may have a case where our object type's projection bound is higher-ranked,
            // but the where clauses we instantiated are not. We can solve this by instantiating
            // the binder at the usage site.
            let proj = self.ecx.instantiate_binder_with_infer(*replacement);
            // FIXME: Technically this folder could be fallible?
            let nested = self
                .ecx
                .eq_and_get_goals(self.param_env, alias_ty, proj.projection_ty)
                .expect("expected to be able to unify goal projection with dyn's projection");
            // FIXME: Technically we could register these too..
            assert!(nested.is_empty(), "did not expect unification to have any nested goals");
            proj.term.ty().unwrap()
        } else {
            ty.super_fold_with(self)
        }
    }
}
