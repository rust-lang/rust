//! Things useful for mapping to/from Chalk and next-trait-solver types.

use base_db::Crate;
use chalk_ir::{
    CanonicalVarKind, CanonicalVarKinds, FnPointer, InferenceVar, Substitution, TyVariableKind,
    WellFormed, cast::Cast, fold::Shift, interner::HasInterner,
};
use hir_def::{
    CallableDefId, ConstParamId, FunctionId, GeneralConstId, LifetimeParamId, TypeAliasId,
    TypeOrConstParamId, TypeParamId, signatures::TraitFlags,
};
use hir_def::{GenericDefId, GenericParamId};
use intern::sym;
use rustc_type_ir::{
    AliasTerm, BoundVar, DebruijnIndex, ExistentialProjection, ExistentialTraitRef, Interner as _,
    OutlivesPredicate, ProjectionPredicate, TypeFoldable, TypeSuperFoldable, TypeVisitable,
    TypeVisitableExt, UniverseIndex, elaborate,
    inherent::{BoundVarLike, Clause as _, IntoKind, PlaceholderLike, SliceLike, Ty as _},
    shift_vars,
    solve::Goal,
};
use salsa::plumbing::FromId;
use salsa::{Id, plumbing::AsId};

use crate::next_solver::BoundConst;
use crate::{
    ConstScalar, ImplTraitId, Interner, MemoryMap,
    db::{
        HirDatabase, InternedClosureId, InternedCoroutineId, InternedLifetimeParamId,
        InternedOpaqueTyId, InternedTypeOrConstParamId,
    },
    from_assoc_type_id, from_chalk_trait_id, from_foreign_def_id,
    mapping::ToChalk,
    next_solver::{
        Binder, ClauseKind, ConstBytes, TraitPredicate, UnevaluatedConst,
        interner::{AdtDef, BoundVarKind, BoundVarKinds, DbInterner},
    },
    to_assoc_type_id, to_chalk_trait_id, to_foreign_def_id,
};
use crate::{
    from_placeholder_idx, lt_from_placeholder_idx, lt_to_placeholder_idx, to_placeholder_idx,
};

use super::{
    BoundExistentialPredicate, BoundExistentialPredicates, BoundRegion, BoundRegionKind, BoundTy,
    BoundTyKind, Canonical, CanonicalVars, Clause, Clauses, Const, Ctor, EarlyParamRegion,
    ErrorGuaranteed, ExistentialPredicate, GenericArg, GenericArgs, ParamConst, ParamEnv, ParamTy,
    Placeholder, PlaceholderConst, PlaceholderRegion, PlaceholderTy, Predicate, PredicateKind,
    Region, SolverDefId, SubtypePredicate, Term, TraitRef, Ty, Tys, ValueConst, VariancesOf,
};

// FIXME: This should urgently go (as soon as we finish the migration off Chalk, that is).
pub fn convert_binder_to_early_binder<'db, T: rustc_type_ir::TypeFoldable<DbInterner<'db>>>(
    interner: DbInterner<'db>,
    def: GenericDefId,
    binder: rustc_type_ir::Binder<DbInterner<'db>, T>,
) -> rustc_type_ir::EarlyBinder<DbInterner<'db>, T> {
    let mut folder = BinderToEarlyBinder {
        interner,
        debruijn: rustc_type_ir::DebruijnIndex::ZERO,
        params: crate::generics::generics(interner.db, def).iter_id().collect(),
    };
    rustc_type_ir::EarlyBinder::bind(binder.skip_binder().fold_with(&mut folder))
}

struct BinderToEarlyBinder<'db> {
    interner: DbInterner<'db>,
    debruijn: rustc_type_ir::DebruijnIndex,
    params: Vec<GenericParamId>,
}

impl<'db> rustc_type_ir::TypeFolder<DbInterner<'db>> for BinderToEarlyBinder<'db> {
    fn cx(&self) -> DbInterner<'db> {
        self.interner
    }

    fn fold_binder<T>(
        &mut self,
        t: rustc_type_ir::Binder<DbInterner<'db>, T>,
    ) -> rustc_type_ir::Binder<DbInterner<'db>, T>
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        self.debruijn.shift_in(1);
        let result = t.super_fold_with(self);
        self.debruijn.shift_out(1);
        result
    }

    fn fold_ty(&mut self, t: Ty<'db>) -> Ty<'db> {
        match t.kind() {
            rustc_type_ir::TyKind::Bound(debruijn, bound_ty) if self.debruijn == debruijn => {
                let var: rustc_type_ir::BoundVar = bound_ty.var();
                let GenericParamId::TypeParamId(id) = self.params[bound_ty.var.as_usize()] else {
                    unreachable!()
                };
                Ty::new(
                    self.cx(),
                    rustc_type_ir::TyKind::Param(ParamTy { index: var.as_u32(), id }),
                )
            }
            _ => t.super_fold_with(self),
        }
    }

    fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
        match r.kind() {
            rustc_type_ir::ReBound(debruijn, bound_region) if self.debruijn == debruijn => {
                let var: rustc_type_ir::BoundVar = bound_region.var();
                let GenericParamId::LifetimeParamId(id) = self.params[bound_region.var.as_usize()]
                else {
                    unreachable!()
                };
                Region::new(
                    self.cx(),
                    rustc_type_ir::RegionKind::ReEarlyParam(EarlyParamRegion {
                        index: var.as_u32(),
                        id,
                    }),
                )
            }
            _ => r,
        }
    }

    fn fold_const(&mut self, c: Const<'db>) -> Const<'db> {
        match c.kind() {
            rustc_type_ir::ConstKind::Bound(debruijn, var) if self.debruijn == debruijn => {
                let GenericParamId::ConstParamId(id) = self.params[var.var.as_usize()] else {
                    unreachable!()
                };
                Const::new(
                    self.cx(),
                    rustc_type_ir::ConstKind::Param(ParamConst { index: var.var.as_u32(), id }),
                )
            }
            _ => c.super_fold_with(self),
        }
    }
}

pub trait ChalkToNextSolver<'db, Out> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Out;
}

pub trait NextSolverToChalk<'db, Out> {
    fn to_chalk(self, interner: DbInterner<'db>) -> Out;
}

impl NextSolverToChalk<'_, chalk_ir::Mutability> for rustc_ast_ir::Mutability {
    fn to_chalk(self, interner: DbInterner<'_>) -> chalk_ir::Mutability {
        match self {
            rustc_ast_ir::Mutability::Not => chalk_ir::Mutability::Not,
            rustc_ast_ir::Mutability::Mut => chalk_ir::Mutability::Mut,
        }
    }
}

impl NextSolverToChalk<'_, chalk_ir::Safety> for crate::next_solver::abi::Safety {
    fn to_chalk(self, interner: DbInterner<'_>) -> chalk_ir::Safety {
        match self {
            crate::next_solver::abi::Safety::Unsafe => chalk_ir::Safety::Unsafe,
            crate::next_solver::abi::Safety::Safe => chalk_ir::Safety::Safe,
        }
    }
}

impl<'db> ChalkToNextSolver<'db, Ty<'db>> for chalk_ir::Ty<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Ty<'db> {
        Ty::new(
            interner,
            match self.kind(Interner) {
                chalk_ir::TyKind::Adt(adt_id, substitution) => {
                    let def = AdtDef::new(adt_id.0, interner);
                    let args = substitution.to_nextsolver(interner);
                    rustc_type_ir::TyKind::Adt(def, args)
                }
                chalk_ir::TyKind::AssociatedType(assoc_type_id, substitution) => {
                    let def_id = SolverDefId::TypeAliasId(from_assoc_type_id(*assoc_type_id));
                    let args: GenericArgs<'db> = substitution.to_nextsolver(interner);
                    let alias_ty = rustc_type_ir::AliasTy::new(interner, def_id, args.iter());
                    rustc_type_ir::TyKind::Alias(rustc_type_ir::AliasTyKind::Projection, alias_ty)
                }
                chalk_ir::TyKind::Scalar(scalar) => match scalar {
                    chalk_ir::Scalar::Bool => rustc_type_ir::TyKind::Bool,
                    chalk_ir::Scalar::Char => rustc_type_ir::TyKind::Char,
                    chalk_ir::Scalar::Int(chalk_ir::IntTy::Isize) => {
                        rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::Isize)
                    }
                    chalk_ir::Scalar::Int(chalk_ir::IntTy::I8) => {
                        rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I8)
                    }
                    chalk_ir::Scalar::Int(chalk_ir::IntTy::I16) => {
                        rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I16)
                    }
                    chalk_ir::Scalar::Int(chalk_ir::IntTy::I32) => {
                        rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I32)
                    }
                    chalk_ir::Scalar::Int(chalk_ir::IntTy::I64) => {
                        rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I64)
                    }
                    chalk_ir::Scalar::Int(chalk_ir::IntTy::I128) => {
                        rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I128)
                    }
                    chalk_ir::Scalar::Uint(chalk_ir::UintTy::Usize) => {
                        rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::Usize)
                    }
                    chalk_ir::Scalar::Uint(chalk_ir::UintTy::U8) => {
                        rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U8)
                    }
                    chalk_ir::Scalar::Uint(chalk_ir::UintTy::U16) => {
                        rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U16)
                    }
                    chalk_ir::Scalar::Uint(chalk_ir::UintTy::U32) => {
                        rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U32)
                    }
                    chalk_ir::Scalar::Uint(chalk_ir::UintTy::U64) => {
                        rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U64)
                    }
                    chalk_ir::Scalar::Uint(chalk_ir::UintTy::U128) => {
                        rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U128)
                    }
                    chalk_ir::Scalar::Float(chalk_ir::FloatTy::F16) => {
                        rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F16)
                    }
                    chalk_ir::Scalar::Float(chalk_ir::FloatTy::F32) => {
                        rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F32)
                    }
                    chalk_ir::Scalar::Float(chalk_ir::FloatTy::F64) => {
                        rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F64)
                    }
                    chalk_ir::Scalar::Float(chalk_ir::FloatTy::F128) => {
                        rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F128)
                    }
                },
                chalk_ir::TyKind::Tuple(_, substitution) => {
                    let args = substitution.to_nextsolver(interner);
                    rustc_type_ir::TyKind::Tuple(args)
                }
                chalk_ir::TyKind::Array(ty, len) => rustc_type_ir::TyKind::Array(
                    ty.to_nextsolver(interner),
                    len.to_nextsolver(interner),
                ),
                chalk_ir::TyKind::Slice(ty) => {
                    rustc_type_ir::TyKind::Slice(ty.to_nextsolver(interner))
                }
                chalk_ir::TyKind::Raw(mutability, ty) => rustc_type_ir::RawPtr(
                    ty.to_nextsolver(interner),
                    mutability.to_nextsolver(interner),
                ),
                chalk_ir::TyKind::Ref(mutability, lifetime, ty) => rustc_type_ir::TyKind::Ref(
                    lifetime.to_nextsolver(interner),
                    ty.to_nextsolver(interner),
                    mutability.to_nextsolver(interner),
                ),
                chalk_ir::TyKind::OpaqueType(def_id, substitution) => {
                    let id: InternedOpaqueTyId = (*def_id).into();
                    let args: GenericArgs<'db> = substitution.to_nextsolver(interner);
                    let alias_ty = rustc_type_ir::AliasTy::new(interner, id.into(), args);
                    rustc_type_ir::TyKind::Alias(rustc_type_ir::AliasTyKind::Opaque, alias_ty)
                }
                chalk_ir::TyKind::FnDef(fn_def_id, substitution) => {
                    let def_id = CallableDefId::from_chalk(interner.db(), *fn_def_id);
                    rustc_type_ir::TyKind::FnDef(
                        def_id.into(),
                        substitution.to_nextsolver(interner),
                    )
                }
                chalk_ir::TyKind::Str => rustc_type_ir::TyKind::Str,
                chalk_ir::TyKind::Never => rustc_type_ir::TyKind::Never,
                chalk_ir::TyKind::Closure(closure_id, substitution) => {
                    let id: InternedClosureId = (*closure_id).into();
                    rustc_type_ir::TyKind::Closure(id.into(), substitution.to_nextsolver(interner))
                }
                chalk_ir::TyKind::Coroutine(coroutine_id, substitution) => {
                    let id: InternedCoroutineId = (*coroutine_id).into();
                    rustc_type_ir::TyKind::Coroutine(
                        id.into(),
                        substitution.to_nextsolver(interner),
                    )
                }
                chalk_ir::TyKind::CoroutineWitness(coroutine_id, substitution) => {
                    let id: InternedCoroutineId = (*coroutine_id).into();
                    rustc_type_ir::TyKind::CoroutineWitness(
                        id.into(),
                        substitution.to_nextsolver(interner),
                    )
                }
                chalk_ir::TyKind::Foreign(foreign_def_id) => rustc_type_ir::TyKind::Foreign(
                    crate::from_foreign_def_id(*foreign_def_id).into(),
                ),
                chalk_ir::TyKind::Error => rustc_type_ir::TyKind::Error(ErrorGuaranteed),
                chalk_ir::TyKind::Dyn(dyn_ty) => {
                    // exists<type> { for<...> ^1.0: ... }
                    let bounds = BoundExistentialPredicates::new_from_iter(
                        interner,
                        dyn_ty.bounds.skip_binders().iter(Interner).filter_map(|pred| {
                            // for<...> ^1.0: ...
                            let (val, binders) = pred.clone().into_value_and_skipped_binders();
                            let bound_vars = binders.to_nextsolver(interner);
                            let clause = match val {
                                chalk_ir::WhereClause::Implemented(trait_ref) => {
                                    let trait_id = from_chalk_trait_id(trait_ref.trait_id);
                                    if interner
                                        .db()
                                        .trait_signature(trait_id)
                                        .flags
                                        .contains(TraitFlags::AUTO)
                                    {
                                        ExistentialPredicate::AutoTrait(trait_id.into())
                                    } else {
                                        let args = GenericArgs::new_from_iter(
                                            interner,
                                            trait_ref
                                                .substitution
                                                .iter(Interner)
                                                .skip(1)
                                                .map(|a| a.clone().shifted_out(Interner).unwrap())
                                                .map(|a| a.to_nextsolver(interner)),
                                        );
                                        let trait_ref = ExistentialTraitRef::new_from_args(
                                            interner, trait_id.into(), args,
                                        );
                                        ExistentialPredicate::Trait(trait_ref)
                                    }
                                }
                                chalk_ir::WhereClause::AliasEq(alias_eq) => {
                                    let (def_id, args) = match &alias_eq.alias {
                                        chalk_ir::AliasTy::Projection(projection) => {
                                            let id =
                                                from_assoc_type_id(projection.associated_ty_id);
                                            let def_id = SolverDefId::TypeAliasId(id);
                                            let generics = interner.generics_of(def_id);
                                            let parent_len = generics.parent_count;
                                            let substs = projection.substitution.iter(Interner).skip(1);

                                            let args = GenericArgs::new_from_iter(
                                                interner,
                                                substs
                                                    .map(|a| {
                                                        a.clone().shifted_out(Interner).unwrap()
                                                    })
                                                    .map(|a| a.to_nextsolver(interner)),
                                            );
                                            (def_id, args)
                                        }
                                        chalk_ir::AliasTy::Opaque(opaque_ty) => {
                                            panic!("Invalid ExistentialPredicate (opaques can't be named).");
                                        }
                                    };
                                    let term = alias_eq
                                        .ty
                                        .clone()
                                        .shifted_out(Interner)
                                        .unwrap()
                                        .to_nextsolver(interner)
                                        .into();
                                    let projection = ExistentialProjection::new_from_args(
                                        interner, def_id, args, term,
                                    );
                                    ExistentialPredicate::Projection(projection)
                                }
                                chalk_ir::WhereClause::LifetimeOutlives(lifetime_outlives) => {
                                    return None;
                                }
                                chalk_ir::WhereClause::TypeOutlives(type_outlives) => return None,
                            };

                            Some(Binder::bind_with_vars(clause, bound_vars))
                        }),
                    );
                    let region = dyn_ty.lifetime.to_nextsolver(interner);
                    rustc_type_ir::TyKind::Dynamic(bounds, region)
                }
                chalk_ir::TyKind::Alias(alias_ty) => match alias_ty {
                    chalk_ir::AliasTy::Projection(projection_ty) => {
                        let def_id = SolverDefId::TypeAliasId(from_assoc_type_id(
                            projection_ty.associated_ty_id,
                        ));
                        let alias_ty = rustc_type_ir::AliasTy::new_from_args(
                            interner,
                            def_id,
                            projection_ty.substitution.to_nextsolver(interner),
                        );
                        rustc_type_ir::TyKind::Alias(
                            rustc_type_ir::AliasTyKind::Projection,
                            alias_ty,
                        )
                    }
                    chalk_ir::AliasTy::Opaque(opaque_ty) => {
                        let id: InternedOpaqueTyId = opaque_ty.opaque_ty_id.into();
                        let def_id = SolverDefId::InternedOpaqueTyId(id);
                        let alias_ty = rustc_type_ir::AliasTy::new_from_args(
                            interner,
                            def_id,
                            opaque_ty.substitution.to_nextsolver(interner),
                        );
                        rustc_type_ir::TyKind::Alias(rustc_type_ir::AliasTyKind::Opaque, alias_ty)
                    }
                },
                chalk_ir::TyKind::Function(fn_pointer) => {
                    let sig_tys = fn_pointer.clone().into_binders(Interner).to_nextsolver(interner);
                    let header = rustc_type_ir::FnHeader {
                        abi: fn_pointer.sig.abi,
                        c_variadic: fn_pointer.sig.variadic,
                        safety: match fn_pointer.sig.safety {
                            chalk_ir::Safety::Safe => super::abi::Safety::Safe,
                            chalk_ir::Safety::Unsafe => super::abi::Safety::Unsafe,
                        },
                    };

                    rustc_type_ir::TyKind::FnPtr(sig_tys, header)
                }
                // The schema here is quite confusing.
                // The new solver, like rustc, uses `Param` and `EarlyBinder` for generic params. It uses `BoundVar`
                // and `Placeholder` together with `Binder` for HRTB, which we mostly don't handle.
                // Chalk uses `Placeholder` for generic params and `BoundVar` quite liberally, and this is quite a
                // problem. `chalk_ir::TyKind::BoundVar` can represent either HRTB or generic params, depending on the
                // context. When returned from signature queries, the outer `Binders` represent the generic params.
                // But there are also inner `Binders` for HRTB.
                // AFAIK there is no way to tell which of the meanings is relevant, so we just use `rustc_type_ir::Bound`
                // here, and hope for the best. If you are working with new solver types, therefore, use the new solver
                // lower queries.
                // Hopefully sooner than later Chalk will be ripped from the codebase and we can avoid that problem.
                // For details about the rustc setup, read: https://rustc-dev-guide.rust-lang.org/generic_parameters_summary.html
                // and the following chapters.
                chalk_ir::TyKind::Placeholder(placeholder_index) => {
                    let (id, index) = from_placeholder_idx(interner.db, *placeholder_index);
                    rustc_type_ir::TyKind::Param(ParamTy {
                        id: TypeParamId::from_unchecked(id),
                        index,
                    })
                }
                chalk_ir::TyKind::BoundVar(bound_var) => rustc_type_ir::TyKind::Bound(
                    bound_var.debruijn.to_nextsolver(interner),
                    BoundTy {
                        var: rustc_type_ir::BoundVar::from_usize(bound_var.index),
                        kind: BoundTyKind::Anon,
                    },
                ),
                chalk_ir::TyKind::InferenceVar(inference_var, ty_variable_kind) => {
                    rustc_type_ir::TyKind::Infer(
                        (*inference_var, *ty_variable_kind).to_nextsolver(interner),
                    )
                }
            },
        )
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::Ty<Interner>> for Ty<'db> {
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::Ty<Interner> {
        convert_ty_for_result(interner, self)
    }
}

impl<'db> ChalkToNextSolver<'db, Region<'db>> for chalk_ir::Lifetime<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Region<'db> {
        Region::new(
            interner,
            match self.data(Interner) {
                chalk_ir::LifetimeData::BoundVar(bound_var) => rustc_type_ir::RegionKind::ReBound(
                    bound_var.debruijn.to_nextsolver(interner),
                    BoundRegion {
                        var: rustc_type_ir::BoundVar::from_u32(bound_var.index as u32),
                        kind: BoundRegionKind::Anon,
                    },
                ),
                chalk_ir::LifetimeData::InferenceVar(inference_var) => {
                    rustc_type_ir::RegionKind::ReVar(rustc_type_ir::RegionVid::from_u32(
                        inference_var.index(),
                    ))
                }
                chalk_ir::LifetimeData::Placeholder(placeholder_index) => {
                    let (id, index) = lt_from_placeholder_idx(interner.db, *placeholder_index);
                    rustc_type_ir::RegionKind::ReEarlyParam(EarlyParamRegion { id, index })
                }
                chalk_ir::LifetimeData::Static => rustc_type_ir::RegionKind::ReStatic,
                chalk_ir::LifetimeData::Erased => rustc_type_ir::RegionKind::ReErased,
                chalk_ir::LifetimeData::Phantom(_, _) => {
                    unreachable!()
                }
                chalk_ir::LifetimeData::Error => {
                    rustc_type_ir::RegionKind::ReError(ErrorGuaranteed)
                }
            },
        )
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::Lifetime<Interner>> for Region<'db> {
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::Lifetime<Interner> {
        convert_region_for_result(interner, self)
    }
}

impl<'db> ChalkToNextSolver<'db, Const<'db>> for chalk_ir::Const<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Const<'db> {
        let data = self.data(Interner);
        Const::new(
            interner,
            match &data.value {
                chalk_ir::ConstValue::BoundVar(bound_var) => rustc_type_ir::ConstKind::Bound(
                    bound_var.debruijn.to_nextsolver(interner),
                    BoundConst { var: rustc_type_ir::BoundVar::from_usize(bound_var.index) },
                ),
                chalk_ir::ConstValue::InferenceVar(inference_var) => {
                    rustc_type_ir::ConstKind::Infer(rustc_type_ir::InferConst::Var(
                        rustc_type_ir::ConstVid::from_u32(inference_var.index()),
                    ))
                }
                chalk_ir::ConstValue::Placeholder(placeholder_index) => {
                    let (id, index) = from_placeholder_idx(interner.db, *placeholder_index);
                    rustc_type_ir::ConstKind::Param(ParamConst {
                        id: ConstParamId::from_unchecked(id),
                        index,
                    })
                }
                chalk_ir::ConstValue::Concrete(concrete_const) => match &concrete_const.interned {
                    ConstScalar::Bytes(bytes, memory) => {
                        rustc_type_ir::ConstKind::Value(ValueConst::new(
                            data.ty.to_nextsolver(interner),
                            ConstBytes(bytes.clone(), memory.clone()),
                        ))
                    }
                    ConstScalar::UnevaluatedConst(c, subst) => {
                        let def = match *c {
                            GeneralConstId::ConstId(id) => SolverDefId::ConstId(id),
                            GeneralConstId::StaticId(id) => SolverDefId::StaticId(id),
                        };
                        let args = subst.to_nextsolver(interner);
                        rustc_type_ir::ConstKind::Unevaluated(UnevaluatedConst::new(def, args))
                    }
                    ConstScalar::Unknown => rustc_type_ir::ConstKind::Error(ErrorGuaranteed),
                },
            },
        )
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::Const<Interner>> for Const<'db> {
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::Const<Interner> {
        convert_const_for_result(interner, self)
    }
}

impl<'db> ChalkToNextSolver<'db, rustc_type_ir::FnSigTys<DbInterner<'db>>>
    for chalk_ir::FnSubst<Interner>
{
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> rustc_type_ir::FnSigTys<DbInterner<'db>> {
        rustc_type_ir::FnSigTys {
            inputs_and_output: Tys::new_from_iter(
                interner,
                self.0.iter(Interner).map(|g| g.assert_ty_ref(Interner).to_nextsolver(interner)),
            ),
        }
    }
}

impl<
    'db,
    U: TypeVisitable<DbInterner<'db>>,
    T: Clone + ChalkToNextSolver<'db, U> + HasInterner<Interner = Interner>,
> ChalkToNextSolver<'db, rustc_type_ir::Binder<DbInterner<'db>, U>> for chalk_ir::Binders<T>
{
    fn to_nextsolver(
        &self,
        interner: DbInterner<'db>,
    ) -> rustc_type_ir::Binder<DbInterner<'db>, U> {
        let (val, binders) = self.clone().into_value_and_skipped_binders();
        rustc_type_ir::Binder::bind_with_vars(
            val.to_nextsolver(interner),
            binders.to_nextsolver(interner),
        )
    }
}

impl<'db, T: NextSolverToChalk<'db, U>, U: HasInterner<Interner = Interner>>
    NextSolverToChalk<'db, chalk_ir::Binders<U>> for rustc_type_ir::Binder<DbInterner<'db>, T>
{
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::Binders<U> {
        chalk_ir::Binders::new(
            self.bound_vars().to_chalk(interner),
            self.skip_binder().to_chalk(interner),
        )
    }
}

impl<'db> ChalkToNextSolver<'db, BoundVarKinds> for chalk_ir::VariableKinds<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> BoundVarKinds {
        BoundVarKinds::new_from_iter(
            interner,
            self.iter(Interner).map(|v| v.to_nextsolver(interner)),
        )
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::VariableKinds<Interner>> for BoundVarKinds {
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::VariableKinds<Interner> {
        chalk_ir::VariableKinds::from_iter(Interner, self.iter().map(|v| v.to_chalk(interner)))
    }
}

impl<'db> ChalkToNextSolver<'db, BoundVarKind> for chalk_ir::VariableKind<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> BoundVarKind {
        match self {
            chalk_ir::VariableKind::Ty(_ty_variable_kind) => BoundVarKind::Ty(BoundTyKind::Anon),
            chalk_ir::VariableKind::Lifetime => BoundVarKind::Region(BoundRegionKind::Anon),
            chalk_ir::VariableKind::Const(_ty) => BoundVarKind::Const,
        }
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::VariableKind<Interner>> for BoundVarKind {
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::VariableKind<Interner> {
        match self {
            BoundVarKind::Ty(_) => chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General),
            BoundVarKind::Region(_) => chalk_ir::VariableKind::Lifetime,
            BoundVarKind::Const => {
                chalk_ir::VariableKind::Const(chalk_ir::TyKind::Error.intern(Interner))
            }
        }
    }
}

impl<'db> ChalkToNextSolver<'db, GenericArg<'db>> for chalk_ir::GenericArg<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> GenericArg<'db> {
        match self.data(Interner) {
            chalk_ir::GenericArgData::Ty(ty) => ty.to_nextsolver(interner).into(),
            chalk_ir::GenericArgData::Lifetime(lifetime) => lifetime.to_nextsolver(interner).into(),
            chalk_ir::GenericArgData::Const(const_) => const_.to_nextsolver(interner).into(),
        }
    }
}

impl<'db> ChalkToNextSolver<'db, GenericArgs<'db>> for chalk_ir::Substitution<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> GenericArgs<'db> {
        GenericArgs::new_from_iter(
            interner,
            self.iter(Interner).map(|arg| -> GenericArg<'db> { arg.to_nextsolver(interner) }),
        )
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::Substitution<Interner>> for GenericArgs<'db> {
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::Substitution<Interner> {
        convert_args_for_result(interner, self.as_slice())
    }
}

impl<'db> ChalkToNextSolver<'db, Tys<'db>> for chalk_ir::Substitution<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Tys<'db> {
        Tys::new_from_iter(
            interner,
            self.iter(Interner).map(|arg| -> Ty<'db> {
                match arg.data(Interner) {
                    chalk_ir::GenericArgData::Ty(ty) => ty.to_nextsolver(interner),
                    chalk_ir::GenericArgData::Lifetime(_) => unreachable!(),
                    chalk_ir::GenericArgData::Const(_) => unreachable!(),
                }
            }),
        )
    }
}

impl<'db> NextSolverToChalk<'db, crate::Substitution> for Tys<'db> {
    fn to_chalk(self, interner: DbInterner<'db>) -> crate::Substitution {
        Substitution::from_iter(
            Interner,
            self.inner().iter().map(|ty| ty.to_chalk(interner).cast(Interner)),
        )
    }
}

impl<'db> ChalkToNextSolver<'db, rustc_type_ir::DebruijnIndex> for chalk_ir::DebruijnIndex {
    fn to_nextsolver(&self, _interner: DbInterner<'db>) -> rustc_type_ir::DebruijnIndex {
        rustc_type_ir::DebruijnIndex::from_u32(self.depth())
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::DebruijnIndex> for rustc_type_ir::DebruijnIndex {
    fn to_chalk(self, _interner: DbInterner<'db>) -> chalk_ir::DebruijnIndex {
        chalk_ir::DebruijnIndex::new(self.index() as u32)
    }
}

impl<'db> ChalkToNextSolver<'db, rustc_type_ir::UniverseIndex> for chalk_ir::UniverseIndex {
    fn to_nextsolver(&self, _interner: DbInterner<'db>) -> rustc_type_ir::UniverseIndex {
        rustc_type_ir::UniverseIndex::from_u32(self.counter as u32)
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::UniverseIndex> for rustc_type_ir::UniverseIndex {
    fn to_chalk(self, _interner: DbInterner<'db>) -> chalk_ir::UniverseIndex {
        chalk_ir::UniverseIndex { counter: self.index() }
    }
}

impl<'db> ChalkToNextSolver<'db, rustc_type_ir::InferTy>
    for (chalk_ir::InferenceVar, chalk_ir::TyVariableKind)
{
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> rustc_type_ir::InferTy {
        match self.1 {
            chalk_ir::TyVariableKind::General => {
                rustc_type_ir::InferTy::TyVar(rustc_type_ir::TyVid::from_u32(self.0.index()))
            }
            chalk_ir::TyVariableKind::Integer => {
                rustc_type_ir::InferTy::IntVar(rustc_type_ir::IntVid::from_u32(self.0.index()))
            }
            chalk_ir::TyVariableKind::Float => {
                rustc_type_ir::InferTy::FloatVar(rustc_type_ir::FloatVid::from_u32(self.0.index()))
            }
        }
    }
}

impl<'db> ChalkToNextSolver<'db, rustc_ast_ir::Mutability> for chalk_ir::Mutability {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> rustc_ast_ir::Mutability {
        match self {
            chalk_ir::Mutability::Mut => rustc_ast_ir::Mutability::Mut,
            chalk_ir::Mutability::Not => rustc_ast_ir::Mutability::Not,
        }
    }
}

impl<'db> ChalkToNextSolver<'db, rustc_type_ir::Variance> for crate::Variance {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> rustc_type_ir::Variance {
        match self {
            crate::Variance::Covariant => rustc_type_ir::Variance::Covariant,
            crate::Variance::Invariant => rustc_type_ir::Variance::Invariant,
            crate::Variance::Contravariant => rustc_type_ir::Variance::Contravariant,
            crate::Variance::Bivariant => rustc_type_ir::Variance::Bivariant,
        }
    }
}

impl<'db> ChalkToNextSolver<'db, rustc_type_ir::Variance> for chalk_ir::Variance {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> rustc_type_ir::Variance {
        match self {
            chalk_ir::Variance::Covariant => rustc_type_ir::Variance::Covariant,
            chalk_ir::Variance::Invariant => rustc_type_ir::Variance::Invariant,
            chalk_ir::Variance::Contravariant => rustc_type_ir::Variance::Contravariant,
        }
    }
}

impl<'db> ChalkToNextSolver<'db, VariancesOf> for chalk_ir::Variances<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> VariancesOf {
        VariancesOf::new_from_iter(
            interner,
            self.as_slice(Interner).iter().map(|v| v.to_nextsolver(interner)),
        )
    }
}

impl<'db> ChalkToNextSolver<'db, Goal<DbInterner<'db>, Predicate<'db>>>
    for chalk_ir::InEnvironment<chalk_ir::Goal<Interner>>
{
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Goal<DbInterner<'db>, Predicate<'db>> {
        Goal::new(
            interner,
            self.environment.to_nextsolver(interner),
            self.goal.to_nextsolver(interner),
        )
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::InEnvironment<chalk_ir::Goal<Interner>>>
    for Goal<DbInterner<'db>, Predicate<'db>>
{
    fn to_chalk(
        self,
        interner: DbInterner<'db>,
    ) -> chalk_ir::InEnvironment<chalk_ir::Goal<Interner>> {
        chalk_ir::InEnvironment {
            environment: self.param_env.to_chalk(interner),
            goal: self.predicate.to_chalk(interner),
        }
    }
}

impl<'db, T: HasInterner<Interner = Interner> + ChalkToNextSolver<'db, U>, U>
    ChalkToNextSolver<'db, Canonical<'db, U>> for chalk_ir::Canonical<T>
{
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Canonical<'db, U> {
        let variables = CanonicalVars::new_from_iter(
            interner,
            self.binders.iter(Interner).map(|k| match &k.kind {
                chalk_ir::VariableKind::Ty(ty_variable_kind) => match ty_variable_kind {
                    // FIXME(next-solver): the info is incorrect, but we have no way to store the information in Chalk.
                    TyVariableKind::General => rustc_type_ir::CanonicalVarKind::Ty {
                        ui: UniverseIndex::ROOT,
                        sub_root: BoundVar::from_u32(0),
                    },
                    TyVariableKind::Integer => rustc_type_ir::CanonicalVarKind::Int,
                    TyVariableKind::Float => rustc_type_ir::CanonicalVarKind::Float,
                },
                chalk_ir::VariableKind::Lifetime => {
                    rustc_type_ir::CanonicalVarKind::Region(UniverseIndex::ROOT)
                }
                chalk_ir::VariableKind::Const(ty) => {
                    rustc_type_ir::CanonicalVarKind::Const(UniverseIndex::ROOT)
                }
            }),
        );
        Canonical {
            max_universe: UniverseIndex::ROOT,
            value: self.value.to_nextsolver(interner),
            variables,
        }
    }
}

impl<'db, T: NextSolverToChalk<'db, U>, U: HasInterner<Interner = Interner>>
    NextSolverToChalk<'db, chalk_ir::Canonical<U>> for Canonical<'db, T>
{
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::Canonical<U> {
        let binders = chalk_ir::CanonicalVarKinds::from_iter(
            Interner,
            self.variables.iter().map(|v| match v {
                rustc_type_ir::CanonicalVarKind::Ty { ui, sub_root: _ } => {
                    chalk_ir::CanonicalVarKind::new(
                        chalk_ir::VariableKind::Ty(TyVariableKind::General),
                        chalk_ir::UniverseIndex { counter: ui.as_usize() },
                    )
                }
                rustc_type_ir::CanonicalVarKind::Int => chalk_ir::CanonicalVarKind::new(
                    chalk_ir::VariableKind::Ty(TyVariableKind::Integer),
                    chalk_ir::UniverseIndex::root(),
                ),
                rustc_type_ir::CanonicalVarKind::Float => chalk_ir::CanonicalVarKind::new(
                    chalk_ir::VariableKind::Ty(TyVariableKind::Float),
                    chalk_ir::UniverseIndex::root(),
                ),
                rustc_type_ir::CanonicalVarKind::Region(ui) => chalk_ir::CanonicalVarKind::new(
                    chalk_ir::VariableKind::Lifetime,
                    chalk_ir::UniverseIndex { counter: ui.as_usize() },
                ),
                rustc_type_ir::CanonicalVarKind::Const(ui) => chalk_ir::CanonicalVarKind::new(
                    chalk_ir::VariableKind::Const(chalk_ir::TyKind::Error.intern(Interner)),
                    chalk_ir::UniverseIndex { counter: ui.as_usize() },
                ),
                rustc_type_ir::CanonicalVarKind::PlaceholderTy(_) => unimplemented!(),
                rustc_type_ir::CanonicalVarKind::PlaceholderRegion(_) => unimplemented!(),
                rustc_type_ir::CanonicalVarKind::PlaceholderConst(_) => unimplemented!(),
            }),
        );
        let value = self.value.to_chalk(interner);
        chalk_ir::Canonical { binders, value }
    }
}

impl<'db> ChalkToNextSolver<'db, Predicate<'db>> for chalk_ir::Goal<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Predicate<'db> {
        match self.data(Interner) {
            chalk_ir::GoalData::Quantified(quantifier_kind, binders) => {
                if !binders.binders.is_empty(Interner) {
                    panic!("Should not be constructed.");
                }
                let (val, _) = binders.clone().into_value_and_skipped_binders();
                val.shifted_out(Interner).unwrap().to_nextsolver(interner)
            }
            chalk_ir::GoalData::Implies(program_clauses, goal) => {
                panic!("Should not be constructed.")
            }
            chalk_ir::GoalData::All(goals) => panic!("Should not be constructed."),
            chalk_ir::GoalData::Not(goal) => panic!("Should not be constructed."),
            chalk_ir::GoalData::EqGoal(eq_goal) => {
                let arg_to_term = |g: &chalk_ir::GenericArg<Interner>| match g.data(Interner) {
                    chalk_ir::GenericArgData::Ty(ty) => Term::Ty(ty.to_nextsolver(interner)),
                    chalk_ir::GenericArgData::Const(const_) => {
                        Term::Const(const_.to_nextsolver(interner))
                    }
                    chalk_ir::GenericArgData::Lifetime(lifetime) => unreachable!(),
                };
                let pred_kind = PredicateKind::AliasRelate(
                    arg_to_term(&eq_goal.a),
                    arg_to_term(&eq_goal.b),
                    rustc_type_ir::AliasRelationDirection::Equate,
                );
                let pred_kind =
                    Binder::bind_with_vars(pred_kind, BoundVarKinds::new_from_iter(interner, []));
                Predicate::new(interner, pred_kind)
            }
            chalk_ir::GoalData::SubtypeGoal(subtype_goal) => {
                let subtype_predicate = SubtypePredicate {
                    a: subtype_goal.a.to_nextsolver(interner),
                    b: subtype_goal.b.to_nextsolver(interner),
                    a_is_expected: true,
                };
                let pred_kind = PredicateKind::Subtype(subtype_predicate);
                let pred_kind = Binder::bind_with_vars(
                    shift_vars(interner, pred_kind, 1),
                    BoundVarKinds::new_from_iter(interner, []),
                );
                Predicate::new(interner, pred_kind)
            }
            chalk_ir::GoalData::DomainGoal(domain_goal) => {
                let pred_kind = domain_goal.to_nextsolver(interner);
                let pred_kind = Binder::bind_with_vars(
                    shift_vars(interner, pred_kind, 1),
                    BoundVarKinds::new_from_iter(interner, []),
                );
                Predicate::new(interner, pred_kind)
            }
            chalk_ir::GoalData::CannotProve => panic!("Should not be constructed."),
        }
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::Goal<Interner>> for Predicate<'db> {
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::Goal<Interner> {
        chalk_ir::Goal::new(Interner, self.kind().skip_binder().to_chalk(interner))
    }
}

impl<'db> NextSolverToChalk<'db, crate::ProjectionTy> for crate::next_solver::AliasTy<'db> {
    fn to_chalk(self, interner: DbInterner<'db>) -> crate::ProjectionTy {
        let SolverDefId::TypeAliasId(assoc_id) = self.def_id else { unreachable!() };
        crate::ProjectionTy {
            associated_ty_id: to_assoc_type_id(assoc_id),
            substitution: self.args.to_chalk(interner),
        }
    }
}

impl<'db> ChalkToNextSolver<'db, ParamEnv<'db>> for chalk_ir::Environment<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> ParamEnv<'db> {
        let clauses = Clauses::new_from_iter(
            interner,
            self.clauses.iter(Interner).map(|c| c.to_nextsolver(interner)),
        );
        let clauses =
            Clauses::new_from_iter(interner, elaborate::elaborate(interner, clauses.iter()));
        ParamEnv { clauses }
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::Environment<Interner>> for ParamEnv<'db> {
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::Environment<Interner> {
        let clauses = chalk_ir::ProgramClauses::from_iter(
            Interner,
            self.clauses.iter().filter_map(|c| -> Option<chalk_ir::ProgramClause<Interner>> {
                c.to_chalk(interner)
            }),
        );
        chalk_ir::Environment { clauses }
    }
}

impl<'db> ChalkToNextSolver<'db, Clause<'db>> for chalk_ir::ProgramClause<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Clause<'db> {
        Clause(Predicate::new(interner, self.data(Interner).0.to_nextsolver(interner)))
    }
}

impl<'db> NextSolverToChalk<'db, Option<chalk_ir::ProgramClause<Interner>>> for Clause<'db> {
    fn to_chalk(self, interner: DbInterner<'db>) -> Option<chalk_ir::ProgramClause<Interner>> {
        let value: chalk_ir::ProgramClauseImplication<Interner> =
            <PredicateKind<'db> as NextSolverToChalk<
                'db,
                Option<chalk_ir::ProgramClauseImplication<Interner>>,
            >>::to_chalk(self.0.kind().skip_binder(), interner)?;
        Some(chalk_ir::ProgramClause::new(
            Interner,
            chalk_ir::ProgramClauseData(chalk_ir::Binders::empty(Interner, value)),
        ))
    }
}

impl<'db> ChalkToNextSolver<'db, PredicateKind<'db>>
    for chalk_ir::ProgramClauseImplication<Interner>
{
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> PredicateKind<'db> {
        assert!(self.conditions.is_empty(Interner));
        assert!(self.constraints.is_empty(Interner));
        self.consequence.to_nextsolver(interner)
    }
}

impl<'db> NextSolverToChalk<'db, Option<chalk_ir::ProgramClauseImplication<Interner>>>
    for PredicateKind<'db>
{
    fn to_chalk(
        self,
        interner: DbInterner<'db>,
    ) -> Option<chalk_ir::ProgramClauseImplication<Interner>> {
        let chalk_ir::GoalData::DomainGoal(consequence) = self.to_chalk(interner) else {
            return None;
        };

        Some(chalk_ir::ProgramClauseImplication {
            consequence,
            conditions: chalk_ir::Goals::empty(Interner),
            constraints: chalk_ir::Constraints::empty(Interner),
            priority: chalk_ir::ClausePriority::High,
        })
    }
}

impl<'db> ChalkToNextSolver<'db, PredicateKind<'db>> for chalk_ir::DomainGoal<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> PredicateKind<'db> {
        match self {
            chalk_ir::DomainGoal::Holds(where_clause) => match where_clause {
                chalk_ir::WhereClause::Implemented(trait_ref) => {
                    let predicate = TraitPredicate {
                        trait_ref: trait_ref.to_nextsolver(interner),
                        polarity: rustc_type_ir::PredicatePolarity::Positive,
                    };
                    PredicateKind::Clause(ClauseKind::Trait(predicate))
                }
                chalk_ir::WhereClause::AliasEq(alias_eq) => match &alias_eq.alias {
                    chalk_ir::AliasTy::Projection(p) => {
                        let def_id =
                            SolverDefId::TypeAliasId(from_assoc_type_id(p.associated_ty_id));
                        let args = p.substitution.to_nextsolver(interner);
                        let term: Ty<'db> = alias_eq.ty.to_nextsolver(interner);
                        let term: Term<'db> = term.into();
                        let predicate = ProjectionPredicate {
                            projection_term: AliasTerm::new_from_args(interner, def_id, args),
                            term,
                        };
                        PredicateKind::Clause(ClauseKind::Projection(predicate))
                    }
                    chalk_ir::AliasTy::Opaque(opaque) => {
                        let id: InternedOpaqueTyId = opaque.opaque_ty_id.into();
                        let def_id = SolverDefId::InternedOpaqueTyId(id);
                        let args = opaque.substitution.to_nextsolver(interner);
                        let term: Ty<'db> = alias_eq.ty.to_nextsolver(interner);
                        let term: Term<'db> = term.into();
                        let opaque_ty = Ty::new(
                            interner,
                            rustc_type_ir::TyKind::Alias(
                                rustc_type_ir::AliasTyKind::Opaque,
                                rustc_type_ir::AliasTy::new_from_args(interner, def_id, args),
                            ),
                        )
                        .into();
                        PredicateKind::AliasRelate(
                            opaque_ty,
                            term,
                            rustc_type_ir::AliasRelationDirection::Equate,
                        )
                    }
                },
                chalk_ir::WhereClause::LifetimeOutlives(lifetime_outlives) => {
                    let predicate = OutlivesPredicate(
                        lifetime_outlives.a.to_nextsolver(interner),
                        lifetime_outlives.b.to_nextsolver(interner),
                    );
                    PredicateKind::Clause(ClauseKind::RegionOutlives(predicate))
                }
                chalk_ir::WhereClause::TypeOutlives(type_outlives) => {
                    let predicate = OutlivesPredicate(
                        type_outlives.ty.to_nextsolver(interner),
                        type_outlives.lifetime.to_nextsolver(interner),
                    );
                    PredicateKind::Clause(ClauseKind::TypeOutlives(predicate))
                }
            },
            chalk_ir::DomainGoal::Normalize(normalize) => {
                let proj_ty = match &normalize.alias {
                    chalk_ir::AliasTy::Projection(proj) => proj,
                    _ => unimplemented!(),
                };
                let args: GenericArgs<'db> = proj_ty.substitution.to_nextsolver(interner);
                let alias = Ty::new(
                    interner,
                    rustc_type_ir::TyKind::Alias(
                        rustc_type_ir::AliasTyKind::Projection,
                        rustc_type_ir::AliasTy::new(
                            interner,
                            from_assoc_type_id(proj_ty.associated_ty_id).into(),
                            args,
                        ),
                    ),
                )
                .into();
                let term = normalize.ty.to_nextsolver(interner).into();
                PredicateKind::AliasRelate(
                    alias,
                    term,
                    rustc_type_ir::AliasRelationDirection::Equate,
                )
            }
            chalk_ir::DomainGoal::WellFormed(well_formed) => {
                let term = match well_formed {
                    WellFormed::Trait(_) => panic!("Should not be constructed."),
                    WellFormed::Ty(ty) => Term::Ty(ty.to_nextsolver(interner)),
                };
                PredicateKind::Clause(rustc_type_ir::ClauseKind::WellFormed(term))
            }
            chalk_ir::DomainGoal::FromEnv(from_env) => match from_env {
                chalk_ir::FromEnv::Trait(trait_ref) => {
                    let predicate = TraitPredicate {
                        trait_ref: trait_ref.to_nextsolver(interner),
                        polarity: rustc_type_ir::PredicatePolarity::Positive,
                    };
                    PredicateKind::Clause(ClauseKind::Trait(predicate))
                }
                chalk_ir::FromEnv::Ty(ty) => PredicateKind::Clause(ClauseKind::WellFormed(
                    Term::Ty(ty.to_nextsolver(interner)),
                )),
            },
            chalk_ir::DomainGoal::IsLocal(ty) => panic!("Should not be constructed."),
            chalk_ir::DomainGoal::IsUpstream(ty) => panic!("Should not be constructed."),
            chalk_ir::DomainGoal::IsFullyVisible(ty) => panic!("Should not be constructed."),
            chalk_ir::DomainGoal::LocalImplAllowed(trait_ref) => {
                panic!("Should not be constructed.")
            }
            chalk_ir::DomainGoal::Compatible => panic!("Should not be constructed."),
            chalk_ir::DomainGoal::DownstreamType(ty) => panic!("Should not be constructed."),
            chalk_ir::DomainGoal::Reveal => panic!("Should not be constructed."),
            chalk_ir::DomainGoal::ObjectSafe(trait_id) => panic!("Should not be constructed."),
        }
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::GoalData<Interner>> for PredicateKind<'db> {
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::GoalData<Interner> {
        match self {
            rustc_type_ir::PredicateKind::Clause(rustc_type_ir::ClauseKind::Trait(trait_pred)) => {
                let trait_ref = trait_pred.trait_ref.to_chalk(interner);
                let where_clause = chalk_ir::WhereClause::Implemented(trait_ref);
                chalk_ir::GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(where_clause))
            }
            rustc_type_ir::PredicateKind::Clause(rustc_type_ir::ClauseKind::Projection(
                proj_predicate,
            )) => {
                let associated_ty_id = match proj_predicate.def_id() {
                    SolverDefId::TypeAliasId(id) => to_assoc_type_id(id),
                    _ => unreachable!(),
                };
                let substitution = proj_predicate.projection_term.args.to_chalk(interner);
                let alias = chalk_ir::AliasTy::Projection(chalk_ir::ProjectionTy {
                    associated_ty_id,
                    substitution,
                });
                let ty = match proj_predicate.term.kind() {
                    rustc_type_ir::TermKind::Ty(ty) => ty,
                    rustc_type_ir::TermKind::Const(_) => unimplemented!(),
                };
                let ty = ty.to_chalk(interner);
                let alias_eq = chalk_ir::AliasEq { alias, ty };
                let where_clause = chalk_ir::WhereClause::AliasEq(alias_eq);
                chalk_ir::GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(where_clause))
            }
            rustc_type_ir::PredicateKind::Clause(rustc_type_ir::ClauseKind::TypeOutlives(
                outlives,
            )) => {
                let lifetime = outlives.1.to_chalk(interner);
                let ty = outlives.0.to_chalk(interner);
                let where_clause =
                    chalk_ir::WhereClause::TypeOutlives(chalk_ir::TypeOutlives { lifetime, ty });
                chalk_ir::GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(where_clause))
            }
            rustc_type_ir::PredicateKind::Clause(rustc_type_ir::ClauseKind::RegionOutlives(
                outlives,
            )) => {
                let a = outlives.0.to_chalk(interner);
                let b = outlives.1.to_chalk(interner);
                let where_clause =
                    chalk_ir::WhereClause::LifetimeOutlives(chalk_ir::LifetimeOutlives { a, b });
                chalk_ir::GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(where_clause))
            }
            rustc_type_ir::PredicateKind::AliasRelate(
                alias_term,
                target_term,
                alias_relation_direction,
            ) => {
                let term_to_generic_arg = |term: Term<'db>| match term {
                    Term::Ty(ty) => chalk_ir::GenericArg::new(
                        Interner,
                        chalk_ir::GenericArgData::Ty(ty.to_chalk(interner)),
                    ),
                    Term::Const(const_) => chalk_ir::GenericArg::new(
                        Interner,
                        chalk_ir::GenericArgData::Const(const_.to_chalk(interner)),
                    ),
                };

                chalk_ir::GoalData::EqGoal(chalk_ir::EqGoal {
                    a: term_to_generic_arg(alias_term),
                    b: term_to_generic_arg(target_term),
                })
            }
            rustc_type_ir::PredicateKind::Clause(_) => unimplemented!(),
            rustc_type_ir::PredicateKind::DynCompatible(_) => unimplemented!(),
            rustc_type_ir::PredicateKind::Subtype(_) => unimplemented!(),
            rustc_type_ir::PredicateKind::Coerce(_) => unimplemented!(),
            rustc_type_ir::PredicateKind::ConstEquate(_, _) => unimplemented!(),
            rustc_type_ir::PredicateKind::Ambiguous => unimplemented!(),
            rustc_type_ir::PredicateKind::NormalizesTo(_) => unimplemented!(),
        }
    }
}

impl<'db> ChalkToNextSolver<'db, TraitRef<'db>> for chalk_ir::TraitRef<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> TraitRef<'db> {
        let args = self.substitution.to_nextsolver(interner);
        TraitRef::new_from_args(interner, from_chalk_trait_id(self.trait_id).into(), args)
    }
}

impl<'db> NextSolverToChalk<'db, chalk_ir::TraitRef<Interner>> for TraitRef<'db> {
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::TraitRef<Interner> {
        let trait_id = to_chalk_trait_id(self.def_id.0);
        let substitution = self.args.to_chalk(interner);
        chalk_ir::TraitRef { trait_id, substitution }
    }
}

impl<'db> ChalkToNextSolver<'db, PredicateKind<'db>> for chalk_ir::WhereClause<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> PredicateKind<'db> {
        match self {
            chalk_ir::WhereClause::Implemented(trait_ref) => {
                let predicate = TraitPredicate {
                    trait_ref: trait_ref.to_nextsolver(interner),
                    polarity: rustc_type_ir::PredicatePolarity::Positive,
                };
                PredicateKind::Clause(ClauseKind::Trait(predicate))
            }
            chalk_ir::WhereClause::AliasEq(alias_eq) => {
                let projection = match &alias_eq.alias {
                    chalk_ir::AliasTy::Projection(p) => p,
                    _ => unimplemented!(),
                };
                let def_id =
                    SolverDefId::TypeAliasId(from_assoc_type_id(projection.associated_ty_id));
                let args = projection.substitution.to_nextsolver(interner);
                let term: Ty<'db> = alias_eq.ty.to_nextsolver(interner);
                let term: Term<'db> = term.into();
                let predicate = ProjectionPredicate {
                    projection_term: AliasTerm::new_from_args(interner, def_id, args),
                    term,
                };
                PredicateKind::Clause(ClauseKind::Projection(predicate))
            }
            chalk_ir::WhereClause::TypeOutlives(type_outlives) => {
                let ty = type_outlives.ty.to_nextsolver(interner);
                let r = type_outlives.lifetime.to_nextsolver(interner);
                PredicateKind::Clause(ClauseKind::TypeOutlives(OutlivesPredicate(ty, r)))
            }
            chalk_ir::WhereClause::LifetimeOutlives(lifetime_outlives) => {
                let a = lifetime_outlives.a.to_nextsolver(interner);
                let b = lifetime_outlives.b.to_nextsolver(interner);
                PredicateKind::Clause(ClauseKind::RegionOutlives(OutlivesPredicate(a, b)))
            }
        }
    }
}

impl<'db, I> NextSolverToChalk<'db, chalk_ir::ConstrainedSubst<Interner>> for I
where
    I: IntoIterator<Item = GenericArg<'db>>,
{
    fn to_chalk(self, interner: DbInterner<'db>) -> chalk_ir::ConstrainedSubst<Interner> {
        chalk_ir::ConstrainedSubst {
            constraints: chalk_ir::Constraints::empty(Interner),
            subst: GenericArgs::new_from_iter(interner, self).to_chalk(interner),
        }
    }
}

impl<'db> NextSolverToChalk<'db, crate::CallableSig> for rustc_type_ir::FnSig<DbInterner<'db>> {
    fn to_chalk(self, interner: DbInterner<'db>) -> crate::CallableSig {
        crate::CallableSig {
            abi: self.abi,
            is_varargs: self.c_variadic,
            safety: match self.safety {
                super::abi::Safety::Safe => chalk_ir::Safety::Safe,
                super::abi::Safety::Unsafe => chalk_ir::Safety::Unsafe,
            },
            params_and_return: triomphe::Arc::from_iter(
                self.inputs_and_output.iter().map(|ty| convert_ty_for_result(interner, ty)),
            ),
        }
    }
}

pub fn convert_canonical_args_for_result<'db>(
    interner: DbInterner<'db>,
    args: Canonical<'db, Vec<GenericArg<'db>>>,
) -> chalk_ir::Canonical<chalk_ir::ConstrainedSubst<Interner>> {
    args.to_chalk(interner)
}

pub fn convert_args_for_result<'db>(
    interner: DbInterner<'db>,
    args: &[GenericArg<'db>],
) -> crate::Substitution {
    let mut substs = Vec::with_capacity(args.len());
    for arg in args {
        match (*arg).kind() {
            rustc_type_ir::GenericArgKind::Type(ty) => {
                let ty = convert_ty_for_result(interner, ty);
                substs.push(chalk_ir::GenericArgData::Ty(ty).intern(Interner));
            }
            rustc_type_ir::GenericArgKind::Lifetime(region) => {
                let lifetime = convert_region_for_result(interner, region);
                substs.push(chalk_ir::GenericArgData::Lifetime(lifetime).intern(Interner));
            }
            rustc_type_ir::GenericArgKind::Const(const_) => {
                substs.push(
                    chalk_ir::GenericArgData::Const(convert_const_for_result(interner, const_))
                        .intern(Interner),
                );
            }
        }
    }
    Substitution::from_iter(Interner, substs)
}

pub fn convert_ty_for_result<'db>(interner: DbInterner<'db>, ty: Ty<'db>) -> crate::Ty {
    use crate::{Scalar, TyKind};
    use chalk_ir::{FloatTy, IntTy, UintTy};
    match ty.kind() {
        rustc_type_ir::TyKind::Bool => TyKind::Scalar(Scalar::Bool),
        rustc_type_ir::TyKind::Char => TyKind::Scalar(Scalar::Char),
        rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I8) => {
            TyKind::Scalar(Scalar::Int(IntTy::I8))
        }
        rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I16) => {
            TyKind::Scalar(Scalar::Int(IntTy::I16))
        }
        rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I32) => {
            TyKind::Scalar(Scalar::Int(IntTy::I32))
        }
        rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I64) => {
            TyKind::Scalar(Scalar::Int(IntTy::I64))
        }
        rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::I128) => {
            TyKind::Scalar(Scalar::Int(IntTy::I128))
        }
        rustc_type_ir::TyKind::Int(rustc_type_ir::IntTy::Isize) => {
            TyKind::Scalar(Scalar::Int(IntTy::Isize))
        }
        rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U8) => {
            TyKind::Scalar(Scalar::Uint(UintTy::U8))
        }
        rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U16) => {
            TyKind::Scalar(Scalar::Uint(UintTy::U16))
        }
        rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U32) => {
            TyKind::Scalar(Scalar::Uint(UintTy::U32))
        }
        rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U64) => {
            TyKind::Scalar(Scalar::Uint(UintTy::U64))
        }
        rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::U128) => {
            TyKind::Scalar(Scalar::Uint(UintTy::U128))
        }
        rustc_type_ir::TyKind::Uint(rustc_type_ir::UintTy::Usize) => {
            TyKind::Scalar(Scalar::Uint(UintTy::Usize))
        }
        rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F16) => {
            TyKind::Scalar(Scalar::Float(FloatTy::F16))
        }
        rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F32) => {
            TyKind::Scalar(Scalar::Float(FloatTy::F32))
        }
        rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F64) => {
            TyKind::Scalar(Scalar::Float(FloatTy::F64))
        }
        rustc_type_ir::TyKind::Float(rustc_type_ir::FloatTy::F128) => {
            TyKind::Scalar(Scalar::Float(FloatTy::F128))
        }
        rustc_type_ir::TyKind::Str => TyKind::Str,
        rustc_type_ir::TyKind::Error(_) => TyKind::Error,
        rustc_type_ir::TyKind::Never => TyKind::Never,

        rustc_type_ir::TyKind::Adt(def, args) => {
            let adt_id = def.inner().id;
            let subst = convert_args_for_result(interner, args.as_slice());
            TyKind::Adt(chalk_ir::AdtId(adt_id), subst)
        }

        rustc_type_ir::TyKind::Infer(infer_ty) => {
            let (var, kind) = match infer_ty {
                rustc_type_ir::InferTy::TyVar(var) => {
                    (InferenceVar::from(var.as_u32()), TyVariableKind::General)
                }
                rustc_type_ir::InferTy::IntVar(var) => {
                    (InferenceVar::from(var.as_u32()), TyVariableKind::Integer)
                }
                rustc_type_ir::InferTy::FloatVar(var) => {
                    (InferenceVar::from(var.as_u32()), TyVariableKind::Float)
                }
                rustc_type_ir::InferTy::FreshFloatTy(..)
                | rustc_type_ir::InferTy::FreshIntTy(..)
                | rustc_type_ir::InferTy::FreshTy(..) => {
                    panic!("Freshening shouldn't happen.")
                }
            };
            TyKind::InferenceVar(var, kind)
        }

        rustc_type_ir::TyKind::Ref(r, ty, mutability) => {
            let mutability = match mutability {
                rustc_ast_ir::Mutability::Mut => chalk_ir::Mutability::Mut,
                rustc_ast_ir::Mutability::Not => chalk_ir::Mutability::Not,
            };
            let r = convert_region_for_result(interner, r);
            let ty = convert_ty_for_result(interner, ty);
            TyKind::Ref(mutability, r, ty)
        }

        rustc_type_ir::TyKind::Tuple(tys) => {
            let size = tys.len();
            let subst = Substitution::from_iter(
                Interner,
                tys.iter().map(|ty| {
                    chalk_ir::GenericArgData::Ty(convert_ty_for_result(interner, ty))
                        .intern(Interner)
                }),
            );
            TyKind::Tuple(size, subst)
        }

        rustc_type_ir::TyKind::Array(ty, const_) => {
            let ty = convert_ty_for_result(interner, ty);
            let const_ = convert_const_for_result(interner, const_);
            TyKind::Array(ty, const_)
        }

        rustc_type_ir::TyKind::Alias(alias_ty_kind, alias_ty) => match alias_ty_kind {
            rustc_type_ir::AliasTyKind::Projection => {
                let assoc_ty_id = match alias_ty.def_id {
                    SolverDefId::TypeAliasId(id) => id,
                    _ => unreachable!(),
                };
                let associated_ty_id = to_assoc_type_id(assoc_ty_id);
                let substitution = convert_args_for_result(interner, alias_ty.args.as_slice());
                TyKind::Alias(crate::AliasTy::Projection(crate::ProjectionTy {
                    associated_ty_id,
                    substitution,
                }))
            }
            rustc_type_ir::AliasTyKind::Opaque => {
                let opaque_ty_id = match alias_ty.def_id {
                    SolverDefId::InternedOpaqueTyId(id) => id,
                    _ => unreachable!(),
                };
                let substitution = convert_args_for_result(interner, alias_ty.args.as_slice());
                TyKind::Alias(chalk_ir::AliasTy::Opaque(chalk_ir::OpaqueTy {
                    opaque_ty_id: opaque_ty_id.into(),
                    substitution,
                }))
            }
            rustc_type_ir::AliasTyKind::Inherent => unimplemented!(),
            rustc_type_ir::AliasTyKind::Free => unimplemented!(),
        },

        // For `Placeholder`, `Bound` and `Param`, see the comment on the reverse conversion.
        rustc_type_ir::TyKind::Placeholder(placeholder) => {
            unimplemented!(
                "A `rustc_type_ir::TyKind::Placeholder` doesn't have a direct \
                correspondence in Chalk, as it represents a universally instantiated `Bound`.\n\
                It therefore feels safer to leave it panicking, but if you hit this panic \
                feel free to do the same as in `rustc_type_ir::TyKind::Bound` here."
            )
        }
        rustc_type_ir::TyKind::Bound(debruijn_index, ty) => TyKind::BoundVar(chalk_ir::BoundVar {
            debruijn: chalk_ir::DebruijnIndex::new(debruijn_index.as_u32()),
            index: ty.var.as_usize(),
        }),
        rustc_type_ir::TyKind::Param(param) => {
            let placeholder = to_placeholder_idx(interner.db, param.id.into(), param.index);
            TyKind::Placeholder(placeholder)
        }

        rustc_type_ir::TyKind::FnPtr(bound_sig, fn_header) => {
            let num_binders = bound_sig.bound_vars().len();
            let sig = chalk_ir::FnSig {
                abi: fn_header.abi,
                safety: match fn_header.safety {
                    crate::next_solver::abi::Safety::Safe => chalk_ir::Safety::Safe,
                    crate::next_solver::abi::Safety::Unsafe => chalk_ir::Safety::Unsafe,
                },
                variadic: fn_header.c_variadic,
            };
            let args = GenericArgs::new_from_iter(
                interner,
                bound_sig.skip_binder().inputs_and_output.iter().map(|a| a.into()),
            );
            let substitution = convert_args_for_result(interner, args.as_slice());
            let substitution = chalk_ir::FnSubst(substitution);
            let fnptr = chalk_ir::FnPointer { num_binders, sig, substitution };
            TyKind::Function(fnptr)
        }

        rustc_type_ir::TyKind::Dynamic(preds, region) => {
            let self_ty = Ty::new_bound(
                interner,
                DebruijnIndex::from_u32(1),
                BoundTy { kind: BoundTyKind::Anon, var: BoundVar::from_u32(0) },
            );
            let bounds = chalk_ir::QuantifiedWhereClauses::from_iter(
                Interner,
                preds.iter().map(|p| {
                    let binders = chalk_ir::VariableKinds::from_iter(
                        Interner,
                        p.bound_vars().iter().map(|b| match b {
                            BoundVarKind::Ty(kind) => {
                                chalk_ir::VariableKind::Ty(TyVariableKind::General)
                            }
                            BoundVarKind::Region(kind) => chalk_ir::VariableKind::Lifetime,
                            BoundVarKind::Const => {
                                chalk_ir::VariableKind::Const(crate::TyKind::Error.intern(Interner))
                            }
                        }),
                    );

                    // Rust and chalk have slightly different
                    // representation for trait objects.
                    //
                    // Chalk uses `for<T0> for<'a> T0: Trait<'a>` while rustc
                    // uses `ExistentialPredicate`s, which do not have a self ty.
                    // We need to shift escaping bound vars by 1 to accommodate
                    // the newly introduced `for<T0>` binder.
                    let p = shift_vars(interner, p, 1);

                    let where_clause = match p.skip_binder() {
                        rustc_type_ir::ExistentialPredicate::Trait(trait_ref) => {
                            let trait_ref = TraitRef::new(
                                interner,
                                trait_ref.def_id,
                                [self_ty.into()].into_iter().chain(trait_ref.args.iter()),
                            );
                            let trait_id = to_chalk_trait_id(trait_ref.def_id.0);
                            let substitution =
                                convert_args_for_result(interner, trait_ref.args.as_slice());
                            let trait_ref = chalk_ir::TraitRef { trait_id, substitution };
                            chalk_ir::WhereClause::Implemented(trait_ref)
                        }
                        rustc_type_ir::ExistentialPredicate::AutoTrait(trait_) => {
                            let trait_id = to_chalk_trait_id(trait_.0);
                            let substitution = chalk_ir::Substitution::from1(
                                Interner,
                                convert_ty_for_result(interner, self_ty),
                            );
                            let trait_ref = chalk_ir::TraitRef { trait_id, substitution };
                            chalk_ir::WhereClause::Implemented(trait_ref)
                        }
                        rustc_type_ir::ExistentialPredicate::Projection(existential_projection) => {
                            let projection = ProjectionPredicate {
                                projection_term: AliasTerm::new(
                                    interner,
                                    existential_projection.def_id,
                                    [self_ty.into()]
                                        .iter()
                                        .chain(existential_projection.args.iter()),
                                ),
                                term: existential_projection.term,
                            };
                            let associated_ty_id = match projection.projection_term.def_id {
                                SolverDefId::TypeAliasId(id) => to_assoc_type_id(id),
                                _ => unreachable!(),
                            };
                            let substitution = convert_args_for_result(
                                interner,
                                projection.projection_term.args.as_slice(),
                            );
                            let alias = chalk_ir::AliasTy::Projection(chalk_ir::ProjectionTy {
                                associated_ty_id,
                                substitution,
                            });
                            let ty = match projection.term {
                                Term::Ty(ty) => ty,
                                _ => unreachable!(),
                            };
                            let ty = convert_ty_for_result(interner, ty);
                            let alias_eq = chalk_ir::AliasEq { alias, ty };
                            chalk_ir::WhereClause::AliasEq(alias_eq)
                        }
                    };
                    chalk_ir::Binders::new(binders, where_clause)
                }),
            );
            let binders = chalk_ir::VariableKinds::from1(
                Interner,
                chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General),
            );
            let bounds = chalk_ir::Binders::new(binders, bounds);
            let dyn_ty =
                chalk_ir::DynTy { bounds, lifetime: convert_region_for_result(interner, region) };
            TyKind::Dyn(dyn_ty)
        }

        rustc_type_ir::TyKind::Slice(ty) => {
            let ty = convert_ty_for_result(interner, ty);
            TyKind::Slice(ty)
        }

        rustc_type_ir::TyKind::Foreign(foreign) => TyKind::Foreign(to_foreign_def_id(foreign.0)),
        rustc_type_ir::TyKind::Pat(_, _) => unimplemented!(),
        rustc_type_ir::TyKind::RawPtr(ty, mutability) => {
            let mutability = match mutability {
                rustc_ast_ir::Mutability::Mut => chalk_ir::Mutability::Mut,
                rustc_ast_ir::Mutability::Not => chalk_ir::Mutability::Not,
            };
            let ty = convert_ty_for_result(interner, ty);
            TyKind::Raw(mutability, ty)
        }
        rustc_type_ir::TyKind::FnDef(def_id, args) => {
            let subst = convert_args_for_result(interner, args.as_slice());
            TyKind::FnDef(def_id.0.to_chalk(interner.db()), subst)
        }

        rustc_type_ir::TyKind::Closure(def_id, args) => {
            let subst = convert_args_for_result(interner, args.as_slice());
            TyKind::Closure(def_id.0.into(), subst)
        }
        rustc_type_ir::TyKind::CoroutineClosure(_, _) => unimplemented!(),
        rustc_type_ir::TyKind::Coroutine(def_id, args) => {
            let subst = convert_args_for_result(interner, args.as_slice());
            TyKind::Coroutine(def_id.0.into(), subst)
        }
        rustc_type_ir::TyKind::CoroutineWitness(def_id, args) => {
            let subst = convert_args_for_result(interner, args.as_slice());
            TyKind::CoroutineWitness(def_id.0.into(), subst)
        }

        rustc_type_ir::TyKind::UnsafeBinder(_) => unimplemented!(),
    }
    .intern(Interner)
}

pub fn convert_const_for_result<'db>(
    interner: DbInterner<'db>,
    const_: Const<'db>,
) -> crate::Const {
    let value: chalk_ir::ConstValue<Interner> = match const_.kind() {
        rustc_type_ir::ConstKind::Infer(rustc_type_ir::InferConst::Var(var)) => {
            chalk_ir::ConstValue::InferenceVar(chalk_ir::InferenceVar::from(var.as_u32()))
        }
        rustc_type_ir::ConstKind::Infer(rustc_type_ir::InferConst::Fresh(fresh)) => {
            panic!("Vars should not be freshened.")
        }
        rustc_type_ir::ConstKind::Param(param) => {
            let placeholder = to_placeholder_idx(interner.db, param.id.into(), param.index);
            chalk_ir::ConstValue::Placeholder(placeholder)
        }
        rustc_type_ir::ConstKind::Bound(debruijn_index, var) => {
            chalk_ir::ConstValue::BoundVar(chalk_ir::BoundVar::new(
                chalk_ir::DebruijnIndex::new(debruijn_index.as_u32()),
                var.var.index(),
            ))
        }
        rustc_type_ir::ConstKind::Placeholder(placeholder_const) => {
            unimplemented!(
                "A `rustc_type_ir::ConstKind::Placeholder` doesn't have a direct \
                correspondence in Chalk, as it represents a universally instantiated `Bound`.\n\
                It therefore feels safer to leave it panicking, but if you hit this panic \
                feel free to do the same as in `rustc_type_ir::ConstKind::Bound` here."
            )
        }
        rustc_type_ir::ConstKind::Unevaluated(unevaluated_const) => {
            let id = match unevaluated_const.def {
                SolverDefId::ConstId(id) => GeneralConstId::ConstId(id),
                SolverDefId::StaticId(id) => GeneralConstId::StaticId(id),
                _ => unreachable!(),
            };
            let subst = convert_args_for_result(interner, unevaluated_const.args.as_slice());
            chalk_ir::ConstValue::Concrete(chalk_ir::ConcreteConst {
                interned: ConstScalar::UnevaluatedConst(id, subst),
            })
        }
        rustc_type_ir::ConstKind::Value(value_const) => {
            let bytes = value_const.value.inner();
            let value = chalk_ir::ConstValue::Concrete(chalk_ir::ConcreteConst {
                // SAFETY: we will never actually use this without a database
                interned: ConstScalar::Bytes(bytes.0.clone(), unsafe {
                    std::mem::transmute::<MemoryMap<'db>, MemoryMap<'static>>(bytes.1.clone())
                }),
            });
            return chalk_ir::ConstData {
                ty: convert_ty_for_result(interner, value_const.ty),
                value,
            }
            .intern(Interner);
        }
        rustc_type_ir::ConstKind::Error(_) => {
            chalk_ir::ConstValue::Concrete(chalk_ir::ConcreteConst {
                interned: ConstScalar::Unknown,
            })
        }
        rustc_type_ir::ConstKind::Expr(_) => unimplemented!(),
    };
    chalk_ir::ConstData { ty: crate::TyKind::Error.intern(Interner), value }.intern(Interner)
}

pub fn convert_region_for_result<'db>(
    interner: DbInterner<'db>,
    region: Region<'db>,
) -> crate::Lifetime {
    let lifetime = match region.kind() {
        rustc_type_ir::RegionKind::ReEarlyParam(early) => {
            let placeholder = lt_to_placeholder_idx(interner.db, early.id, early.index);
            chalk_ir::LifetimeData::Placeholder(placeholder)
        }
        rustc_type_ir::RegionKind::ReBound(db, bound) => {
            chalk_ir::LifetimeData::BoundVar(chalk_ir::BoundVar::new(
                chalk_ir::DebruijnIndex::new(db.as_u32()),
                bound.var.as_usize(),
            ))
        }
        rustc_type_ir::RegionKind::RePlaceholder(placeholder) => unimplemented!(
            "A `rustc_type_ir::RegionKind::RePlaceholder` doesn't have a direct \
            correspondence in Chalk, as it represents a universally instantiated `Bound`.\n\
            It therefore feels safer to leave it panicking, but if you hit this panic \
            feel free to do the same as in `rustc_type_ir::RegionKind::ReBound` here."
        ),
        rustc_type_ir::RegionKind::ReLateParam(_) => unimplemented!(),
        rustc_type_ir::RegionKind::ReStatic => chalk_ir::LifetimeData::Static,
        rustc_type_ir::RegionKind::ReVar(vid) => {
            chalk_ir::LifetimeData::InferenceVar(chalk_ir::InferenceVar::from(vid.as_u32()))
        }
        rustc_type_ir::RegionKind::ReErased => chalk_ir::LifetimeData::Erased,
        rustc_type_ir::RegionKind::ReError(_) => chalk_ir::LifetimeData::Error,
    };
    chalk_ir::Lifetime::new(Interner, lifetime)
}

pub trait InferenceVarExt {
    fn to_vid(self) -> rustc_type_ir::TyVid;
    fn from_vid(vid: rustc_type_ir::TyVid) -> InferenceVar;
}

impl InferenceVarExt for InferenceVar {
    fn to_vid(self) -> rustc_type_ir::TyVid {
        rustc_type_ir::TyVid::from_u32(self.index())
    }
    fn from_vid(vid: rustc_type_ir::TyVid) -> InferenceVar {
        InferenceVar::from(vid.as_u32())
    }
}
