use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{Movability, Mutability};
use rustc_infer::traits::query::NoSolution;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable};

use crate::solve::EvalCtxt;

// Calculates the constituent types of a type for `auto trait` purposes.
//
// For types with an "existential" binder, i.e. generator witnesses, we also
// instantiate the binder with placeholders eagerly.
pub(super) fn instantiate_constituent_tys_for_auto_trait<'tcx>(
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
        | ty::Str
        | ty::Error(_)
        | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
        | ty::Never
        | ty::Char => Ok(vec![]),

        ty::Dynamic(..)
        | ty::Param(..)
        | ty::Foreign(..)
        | ty::Alias(ty::Projection, ..)
        | ty::Placeholder(..) => Err(NoSolution),

        ty::Bound(..)
        | ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
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

        ty::GeneratorWitnessMIR(..) => todo!(),

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

pub(super) fn instantiate_constituent_tys_for_sized_trait<'tcx>(
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

pub(super) fn instantiate_constituent_tys_for_copy_clone_trait<'tcx>(
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

        ty::GeneratorWitnessMIR(..) => todo!(),
    }
}

// Returns a binder of the tupled inputs types and output type from a builtin callable type.
pub(crate) fn extract_tupled_inputs_and_output_from_callable<'tcx>(
    tcx: TyCtxt<'tcx>,
    self_ty: Ty<'tcx>,
    goal_kind: ty::ClosureKind,
) -> Result<Option<ty::Binder<'tcx, (Ty<'tcx>, Ty<'tcx>)>>, NoSolution> {
    match *self_ty.kind() {
        ty::FnDef(def_id, substs) => Ok(Some(
            tcx.fn_sig(def_id)
                .subst(tcx, substs)
                .map_bound(|sig| (tcx.intern_tup(sig.inputs()), sig.output())),
        )),
        ty::FnPtr(sig) => {
            Ok(Some(sig.map_bound(|sig| (tcx.intern_tup(sig.inputs()), sig.output()))))
        }
        ty::Closure(_, substs) => {
            let closure_substs = substs.as_closure();
            match closure_substs.kind_ty().to_opt_closure_kind() {
                Some(closure_kind) if closure_kind.extends(goal_kind) => {}
                None => return Ok(None),
                _ => return Err(NoSolution),
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

pub(crate) fn predicates_for_object_candidate<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    object_bound: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
) -> Vec<ty::Predicate<'tcx>> {
    let mut requirements = vec![];
    requirements.extend(
        tcx.super_predicates_of(trait_ref.def_id).instantiate(tcx, trait_ref.substs).predicates,
    );
    for item in tcx.associated_items(trait_ref.def_id).in_definition_order() {
        if item.kind == ty::AssocKind::Type {
            requirements.extend(tcx.item_bounds(item.def_id).subst(tcx, trait_ref.substs));
        }
    }

    let mut replace_projection_with = FxHashMap::default();
    for bound in object_bound {
        let bound = bound.no_bound_vars().expect("higher-ranked projections not supported, yet");
        if let ty::ExistentialPredicate::Projection(proj) = bound {
            let proj = proj.with_self_ty(tcx, trait_ref.self_ty());
            let old_ty = replace_projection_with.insert(
                proj.projection_ty,
                proj.term.ty().expect("expected only types in dyn right now"),
            );
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

    requirements.fold_with(&mut ReplaceProjectionWith { tcx, mapping: replace_projection_with })
}

struct ReplaceProjectionWith<'tcx> {
    tcx: TyCtxt<'tcx>,
    mapping: FxHashMap<ty::AliasTy<'tcx>, Ty<'tcx>>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ReplaceProjectionWith<'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Alias(ty::Projection, alias_ty) = *ty.kind()
            && let Some(replacement) = self.mapping.get(&alias_ty)
        {
            *replacement
        } else {
            ty.super_fold_with(self)
        }
    }
}
