use rustc_hir::{Movability, Mutability};
use rustc_infer::{infer::InferCtxt, traits::query::NoSolution};
use rustc_middle::ty::{self, Ty};

// Calculates the constituent types of a type for `auto trait` purposes.
//
// For types with an "existential" binder, i.e. generator witnesses, we also
// instantiate the binder with placeholders eagerly.
pub(super) fn instantiate_constituent_tys_for_auto_trait<'tcx>(
    infcx: &InferCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Result<Vec<Ty<'tcx>>, NoSolution> {
    let tcx = infcx.tcx;
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

        ty::Placeholder(..)
        | ty::Dynamic(..)
        | ty::Param(..)
        | ty::Foreign(..)
        | ty::Alias(ty::Projection, ..)
        | ty::Bound(..)
        | ty::Infer(ty::TyVar(_)) => {
            // FIXME: Do we need to mark anything as ambiguous here? Yeah?
            Err(NoSolution)
        }

        ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => bug!(),

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

        ty::GeneratorWitness(types) => {
            Ok(infcx.replace_bound_vars_with_placeholders(types).to_vec())
        }

        // For `PhantomData<T>`, we pass `T`.
        ty::Adt(def, substs) if def.is_phantom_data() => Ok(vec![substs.type_at(0)]),

        ty::Adt(def, substs) => Ok(def.all_fields().map(|f| f.ty(tcx, substs)).collect()),

        ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) => {
            // We can resolve the `impl Trait` to its concrete type,
            // which enforces a DAG between the functions requiring
            // the auto trait bounds in question.
            Ok(vec![tcx.bound_type_of(def_id).subst(tcx, substs)])
        }
    }
}

pub(super) fn instantiate_constituent_tys_for_sized_trait<'tcx>(
    infcx: &InferCtxt<'tcx>,
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
        | ty::Param(_) => Err(NoSolution),

        ty::Infer(ty::TyVar(_)) => bug!("FIXME: ambiguous"),

        ty::Placeholder(..)
        | ty::Bound(..)
        | ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => bug!(),

        ty::Tuple(tys) => Ok(tys.to_vec()),

        ty::Adt(def, substs) => {
            let sized_crit = def.sized_constraint(infcx.tcx);
            Ok(sized_crit
                .0
                .iter()
                .map(|ty| sized_crit.rebind(*ty).subst(infcx.tcx, substs))
                .collect())
        }
    }
}

pub(super) fn instantiate_constituent_tys_for_copy_clone_trait<'tcx>(
    infcx: &InferCtxt<'tcx>,
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
        | ty::Param(_) => Err(NoSolution),

        ty::Infer(ty::TyVar(_)) => bug!("FIXME: ambiguous"),

        ty::Placeholder(..)
        | ty::Bound(..)
        | ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => bug!(),

        ty::Tuple(tys) => Ok(tys.to_vec()),

        ty::Closure(_, substs) => Ok(vec![substs.as_closure().tupled_upvars_ty()]),

        ty::Generator(_, substs, Movability::Movable) => {
            if infcx.tcx.features().generator_clone {
                let generator = substs.as_generator();
                Ok(vec![generator.tupled_upvars_ty(), generator.witness()])
            } else {
                Err(NoSolution)
            }
        }

        ty::GeneratorWitness(types) => {
            Ok(infcx.replace_bound_vars_with_placeholders(types).to_vec())
        }
    }
}
