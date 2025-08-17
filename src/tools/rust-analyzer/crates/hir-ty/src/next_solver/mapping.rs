//! Things useful for mapping to/from Chalk and next-trait-solver types.

use base_db::Crate;
use chalk_ir::{
    CanonicalVarKind, CanonicalVarKinds, ForeignDefId, InferenceVar, Substitution, TyVariableKind,
    WellFormed, fold::Shift, interner::HasInterner,
};
use hir_def::{
    CallableDefId, ConstParamId, FunctionId, GeneralConstId, LifetimeParamId, TypeAliasId,
    TypeOrConstParamId, TypeParamId, signatures::TraitFlags,
};
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

use crate::{
    ConcreteConst, ConstScalar, ImplTraitId, Interner, MemoryMap,
    db::{
        HirDatabase, InternedClosureId, InternedCoroutineId, InternedLifetimeParamId,
        InternedOpaqueTyId, InternedTypeOrConstParamId,
    },
    from_assoc_type_id, from_chalk_trait_id,
    mapping::ToChalk,
    next_solver::{
        Binder, ClauseKind, ConstBytes, TraitPredicate, UnevaluatedConst,
        interner::{AdtDef, BoundVarKind, BoundVarKinds, DbInterner},
    },
    to_assoc_type_id, to_chalk_trait_id, to_foreign_def_id,
};

use super::{
    BoundExistentialPredicate, BoundExistentialPredicates, BoundRegion, BoundRegionKind, BoundTy,
    BoundTyKind, Canonical, CanonicalVars, Clause, Clauses, Const, Ctor, EarlyParamRegion,
    ErrorGuaranteed, ExistentialPredicate, GenericArg, GenericArgs, ParamConst, ParamEnv, ParamTy,
    Placeholder, PlaceholderConst, PlaceholderRegion, PlaceholderTy, Predicate, PredicateKind,
    Region, SolverDefId, SubtypePredicate, Term, TraitRef, Ty, Tys, ValueConst, VariancesOf,
};

pub fn to_placeholder_idx<T: Clone + std::fmt::Debug>(
    db: &dyn HirDatabase,
    id: TypeOrConstParamId,
    map: impl Fn(BoundVar) -> T,
) -> Placeholder<T> {
    let interned_id = InternedTypeOrConstParamId::new(db, id);
    Placeholder {
        universe: UniverseIndex::ZERO,
        bound: map(BoundVar::from_usize(interned_id.as_id().index() as usize)),
    }
}

pub fn bound_var_to_type_or_const_param_idx(
    db: &dyn HirDatabase,
    var: rustc_type_ir::BoundVar,
) -> TypeOrConstParamId {
    // SAFETY: We cannot really encapsulate this unfortunately, so just hope this is sound.
    let interned_id = InternedTypeOrConstParamId::from_id(unsafe { Id::from_index(var.as_u32()) });
    interned_id.loc(db)
}

pub fn bound_var_to_lifetime_idx(
    db: &dyn HirDatabase,
    var: rustc_type_ir::BoundVar,
) -> LifetimeParamId {
    // SAFETY: We cannot really encapsulate this unfortunately, so just hope this is sound.
    let interned_id = InternedLifetimeParamId::from_id(unsafe { Id::from_index(var.as_u32()) });
    interned_id.loc(db)
}

pub fn convert_binder_to_early_binder<'db, T: rustc_type_ir::TypeFoldable<DbInterner<'db>>>(
    interner: DbInterner<'db>,
    binder: rustc_type_ir::Binder<DbInterner<'db>, T>,
) -> rustc_type_ir::EarlyBinder<DbInterner<'db>, T> {
    let mut folder = BinderToEarlyBinder { interner, debruijn: rustc_type_ir::DebruijnIndex::ZERO };
    rustc_type_ir::EarlyBinder::bind(binder.skip_binder().fold_with(&mut folder))
}

struct BinderToEarlyBinder<'db> {
    interner: DbInterner<'db>,
    debruijn: rustc_type_ir::DebruijnIndex,
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
                Ty::new(self.cx(), rustc_type_ir::TyKind::Param(ParamTy { index: var.as_u32() }))
            }
            _ => t.super_fold_with(self),
        }
    }

    fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
        match r.kind() {
            rustc_type_ir::ReBound(debruijn, bound_region) if self.debruijn == debruijn => {
                let var: rustc_type_ir::BoundVar = bound_region.var();
                Region::new(
                    self.cx(),
                    rustc_type_ir::RegionKind::ReEarlyParam(EarlyParamRegion {
                        index: var.as_u32(),
                    }),
                )
            }
            _ => r,
        }
    }

    fn fold_const(&mut self, c: Const<'db>) -> Const<'db> {
        match c.kind() {
            rustc_type_ir::ConstKind::Bound(debruijn, var) if self.debruijn == debruijn => {
                Const::new(
                    self.cx(),
                    rustc_type_ir::ConstKind::Param(ParamConst { index: var.as_u32() }),
                )
            }
            _ => c.super_fold_with(self),
        }
    }
}

pub trait ChalkToNextSolver<'db, Out> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Out;
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
                    let id: SolverDefId = match def_id {
                        CallableDefId::FunctionId(id) => id.into(),
                        CallableDefId::StructId(id) => SolverDefId::Ctor(Ctor::Struct(id)),
                        CallableDefId::EnumVariantId(id) => SolverDefId::Ctor(Ctor::Enum(id)),
                    };
                    rustc_type_ir::TyKind::FnDef(id, substitution.to_nextsolver(interner))
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
                    SolverDefId::ForeignId(crate::from_foreign_def_id(*foreign_def_id)),
                ),
                chalk_ir::TyKind::Error => rustc_type_ir::TyKind::Error(ErrorGuaranteed),
                chalk_ir::TyKind::Placeholder(placeholder_index) => {
                    rustc_type_ir::TyKind::Placeholder(PlaceholderTy::new_anon(
                        placeholder_index.ui.to_nextsolver(interner),
                        rustc_type_ir::BoundVar::from_usize(placeholder_index.idx),
                    ))
                }
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
                                        ExistentialPredicate::AutoTrait(SolverDefId::TraitId(
                                            trait_id,
                                        ))
                                    } else {
                                        let def_id = SolverDefId::TraitId(trait_id);
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
                                            interner, def_id, args,
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
                    let kind = rustc_type_ir::DynKind::Dyn;
                    rustc_type_ir::TyKind::Dynamic(bounds, region, kind)
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
                    rustc_type_ir::RegionKind::RePlaceholder(PlaceholderRegion::new_anon(
                        rustc_type_ir::UniverseIndex::from_u32(placeholder_index.ui.counter as u32),
                        rustc_type_ir::BoundVar::from_u32(placeholder_index.idx as u32),
                    ))
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

impl<'db> ChalkToNextSolver<'db, Const<'db>> for chalk_ir::Const<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Const<'db> {
        let data = self.data(Interner);
        Const::new(
            interner,
            match &data.value {
                chalk_ir::ConstValue::BoundVar(bound_var) => rustc_type_ir::ConstKind::Bound(
                    bound_var.debruijn.to_nextsolver(interner),
                    rustc_type_ir::BoundVar::from_usize(bound_var.index),
                ),
                chalk_ir::ConstValue::InferenceVar(inference_var) => {
                    rustc_type_ir::ConstKind::Infer(rustc_type_ir::InferConst::Var(
                        rustc_type_ir::ConstVid::from_u32(inference_var.index()),
                    ))
                }
                chalk_ir::ConstValue::Placeholder(placeholder_index) => {
                    rustc_type_ir::ConstKind::Placeholder(PlaceholderConst::new(
                        placeholder_index.ui.to_nextsolver(interner),
                        rustc_type_ir::BoundVar::from_usize(placeholder_index.idx),
                    ))
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

impl<'db> ChalkToNextSolver<'db, BoundVarKinds> for chalk_ir::VariableKinds<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> BoundVarKinds {
        BoundVarKinds::new_from_iter(
            interner,
            self.iter(Interner).map(|v| v.to_nextsolver(interner)),
        )
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

impl<'db> ChalkToNextSolver<'db, rustc_type_ir::DebruijnIndex> for chalk_ir::DebruijnIndex {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> rustc_type_ir::DebruijnIndex {
        rustc_type_ir::DebruijnIndex::from_u32(self.depth())
    }
}

impl<'db> ChalkToNextSolver<'db, rustc_type_ir::UniverseIndex> for chalk_ir::UniverseIndex {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> rustc_type_ir::UniverseIndex {
        rustc_type_ir::UniverseIndex::from_u32(self.counter as u32)
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

impl<'db> ChalkToNextSolver<'db, Canonical<'db, Goal<DbInterner<'db>, Predicate<'db>>>>
    for chalk_ir::Canonical<chalk_ir::InEnvironment<chalk_ir::Goal<Interner>>>
{
    fn to_nextsolver(
        &self,
        interner: DbInterner<'db>,
    ) -> Canonical<'db, Goal<DbInterner<'db>, Predicate<'db>>> {
        let param_env = self.value.environment.to_nextsolver(interner);
        let variables = CanonicalVars::new_from_iter(
            interner,
            self.binders.iter(Interner).map(|k| match &k.kind {
                chalk_ir::VariableKind::Ty(ty_variable_kind) => match ty_variable_kind {
                    TyVariableKind::General => rustc_type_ir::CanonicalVarKind::Ty(
                        rustc_type_ir::CanonicalTyVarKind::General(UniverseIndex::ROOT),
                    ),
                    TyVariableKind::Integer => {
                        rustc_type_ir::CanonicalVarKind::Ty(rustc_type_ir::CanonicalTyVarKind::Int)
                    }
                    TyVariableKind::Float => rustc_type_ir::CanonicalVarKind::Ty(
                        rustc_type_ir::CanonicalTyVarKind::Float,
                    ),
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
            value: Goal::new(interner, param_env, self.value.goal.to_nextsolver(interner)),
            variables,
        }
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
            chalk_ir::GoalData::EqGoal(eq_goal) => panic!("Should not be constructed."),
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

impl<'db> ChalkToNextSolver<'db, Clause<'db>> for chalk_ir::ProgramClause<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> Clause<'db> {
        Clause(Predicate::new(interner, self.data(Interner).0.to_nextsolver(interner)))
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
                let alias = rustc_type_ir::AliasTerm::new(
                    interner,
                    from_assoc_type_id(proj_ty.associated_ty_id).into(),
                    args,
                );
                let term = normalize.ty.to_nextsolver(interner).into();
                let normalizes_to = rustc_type_ir::NormalizesTo { alias, term };
                PredicateKind::NormalizesTo(normalizes_to)
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

impl<'db> ChalkToNextSolver<'db, TraitRef<'db>> for chalk_ir::TraitRef<Interner> {
    fn to_nextsolver(&self, interner: DbInterner<'db>) -> TraitRef<'db> {
        let args = self.substitution.to_nextsolver(interner);
        TraitRef::new_from_args(
            interner,
            SolverDefId::TraitId(from_chalk_trait_id(self.trait_id)),
            args,
        )
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

pub fn convert_canonical_args_for_result<'db>(
    interner: DbInterner<'db>,
    args: Canonical<'db, Vec<GenericArg<'db>>>,
) -> chalk_ir::Canonical<chalk_ir::ConstrainedSubst<Interner>> {
    let Canonical { value, variables, max_universe } = args;
    let binders = CanonicalVarKinds::from_iter(
        Interner,
        variables.iter().map(|v| match v {
            rustc_type_ir::CanonicalVarKind::Ty(rustc_type_ir::CanonicalTyVarKind::General(_)) => {
                CanonicalVarKind::new(
                    chalk_ir::VariableKind::Ty(TyVariableKind::General),
                    chalk_ir::UniverseIndex::ROOT,
                )
            }
            rustc_type_ir::CanonicalVarKind::Ty(rustc_type_ir::CanonicalTyVarKind::Int) => {
                CanonicalVarKind::new(
                    chalk_ir::VariableKind::Ty(TyVariableKind::Integer),
                    chalk_ir::UniverseIndex::ROOT,
                )
            }
            rustc_type_ir::CanonicalVarKind::Ty(rustc_type_ir::CanonicalTyVarKind::Float) => {
                CanonicalVarKind::new(
                    chalk_ir::VariableKind::Ty(TyVariableKind::Float),
                    chalk_ir::UniverseIndex::ROOT,
                )
            }
            rustc_type_ir::CanonicalVarKind::Region(universe_index) => CanonicalVarKind::new(
                chalk_ir::VariableKind::Lifetime,
                chalk_ir::UniverseIndex::ROOT,
            ),
            rustc_type_ir::CanonicalVarKind::Const(universe_index) => CanonicalVarKind::new(
                chalk_ir::VariableKind::Const(crate::TyKind::Error.intern(Interner)),
                chalk_ir::UniverseIndex::ROOT,
            ),
            rustc_type_ir::CanonicalVarKind::PlaceholderTy(_) => unimplemented!(),
            rustc_type_ir::CanonicalVarKind::PlaceholderRegion(_) => unimplemented!(),
            rustc_type_ir::CanonicalVarKind::PlaceholderConst(_) => unimplemented!(),
        }),
    );
    chalk_ir::Canonical {
        binders,
        value: chalk_ir::ConstrainedSubst {
            constraints: chalk_ir::Constraints::empty(Interner),
            subst: convert_args_for_result(interner, &value),
        },
    }
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
                let lifetime = convert_region_for_result(region);
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

pub(crate) fn convert_ty_for_result<'db>(interner: DbInterner<'db>, ty: Ty<'db>) -> crate::Ty {
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
            let r = convert_region_for_result(r);
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
                TyKind::AssociatedType(associated_ty_id, substitution)
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

        rustc_type_ir::TyKind::Placeholder(placeholder) => {
            let ui = chalk_ir::UniverseIndex { counter: placeholder.universe.as_usize() };
            let placeholder_index =
                chalk_ir::PlaceholderIndex { idx: placeholder.bound.var.as_usize(), ui };
            TyKind::Placeholder(placeholder_index)
        }

        rustc_type_ir::TyKind::Bound(debruijn_index, ty) => TyKind::BoundVar(chalk_ir::BoundVar {
            debruijn: chalk_ir::DebruijnIndex::new(debruijn_index.as_u32()),
            index: ty.var.as_usize(),
        }),

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

        rustc_type_ir::TyKind::Dynamic(preds, region, dyn_kind) => {
            assert!(matches!(dyn_kind, rustc_type_ir::DynKind::Dyn));
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
                            let trait_id = match trait_ref.def_id {
                                SolverDefId::TraitId(id) => to_chalk_trait_id(id),
                                _ => unreachable!(),
                            };
                            let substitution =
                                convert_args_for_result(interner, trait_ref.args.as_slice());
                            let trait_ref = chalk_ir::TraitRef { trait_id, substitution };
                            chalk_ir::WhereClause::Implemented(trait_ref)
                        }
                        rustc_type_ir::ExistentialPredicate::AutoTrait(trait_) => {
                            let trait_id = match trait_ {
                                SolverDefId::TraitId(id) => to_chalk_trait_id(id),
                                _ => unreachable!(),
                            };
                            let substitution = chalk_ir::Substitution::empty(Interner);
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
            let dyn_ty = chalk_ir::DynTy { bounds, lifetime: convert_region_for_result(region) };
            TyKind::Dyn(dyn_ty)
        }

        rustc_type_ir::TyKind::Slice(ty) => {
            let ty = convert_ty_for_result(interner, ty);
            TyKind::Slice(ty)
        }

        rustc_type_ir::TyKind::Foreign(foreign) => {
            let def_id = match foreign {
                SolverDefId::ForeignId(id) => id,
                _ => unreachable!(),
            };
            TyKind::Foreign(to_foreign_def_id(def_id))
        }
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
            let id = match def_id {
                SolverDefId::FunctionId(id) => CallableDefId::FunctionId(id),
                SolverDefId::Ctor(Ctor::Struct(id)) => CallableDefId::StructId(id),
                SolverDefId::Ctor(Ctor::Enum(id)) => CallableDefId::EnumVariantId(id),
                _ => unreachable!(),
            };
            let subst = convert_args_for_result(interner, args.as_slice());
            TyKind::FnDef(id.to_chalk(interner.db()), subst)
        }

        rustc_type_ir::TyKind::Closure(def_id, args) => {
            let id = match def_id {
                SolverDefId::InternedClosureId(id) => id,
                _ => unreachable!(),
            };
            let subst = convert_args_for_result(interner, args.as_slice());
            TyKind::Closure(id.into(), subst)
        }
        rustc_type_ir::TyKind::CoroutineClosure(_, _) => unimplemented!(),
        rustc_type_ir::TyKind::Coroutine(def_id, args) => {
            let id = match def_id {
                SolverDefId::InternedCoroutineId(id) => id,
                _ => unreachable!(),
            };
            let subst = convert_args_for_result(interner, args.as_slice());
            TyKind::Coroutine(id.into(), subst)
        }
        rustc_type_ir::TyKind::CoroutineWitness(def_id, args) => {
            let id = match def_id {
                SolverDefId::InternedCoroutineId(id) => id,
                _ => unreachable!(),
            };
            let subst = convert_args_for_result(interner, args.as_slice());
            TyKind::CoroutineWitness(id.into(), subst)
        }

        rustc_type_ir::TyKind::Param(_) => unimplemented!(),
        rustc_type_ir::TyKind::UnsafeBinder(_) => unimplemented!(),
    }
    .intern(Interner)
}

pub fn convert_const_for_result<'db>(
    interner: DbInterner<'db>,
    const_: Const<'db>,
) -> crate::Const {
    let value: chalk_ir::ConstValue<Interner> = match const_.kind() {
        rustc_type_ir::ConstKind::Param(_) => unimplemented!(),
        rustc_type_ir::ConstKind::Infer(rustc_type_ir::InferConst::Var(var)) => {
            chalk_ir::ConstValue::InferenceVar(chalk_ir::InferenceVar::from(var.as_u32()))
        }
        rustc_type_ir::ConstKind::Infer(rustc_type_ir::InferConst::Fresh(fresh)) => {
            panic!("Vars should not be freshened.")
        }
        rustc_type_ir::ConstKind::Bound(debruijn_index, var) => {
            chalk_ir::ConstValue::BoundVar(chalk_ir::BoundVar::new(
                chalk_ir::DebruijnIndex::new(debruijn_index.as_u32()),
                var.index(),
            ))
        }
        rustc_type_ir::ConstKind::Placeholder(placeholder_const) => {
            chalk_ir::ConstValue::Placeholder(chalk_ir::PlaceholderIndex {
                ui: chalk_ir::UniverseIndex { counter: placeholder_const.universe.as_usize() },
                idx: placeholder_const.bound.as_usize(),
            })
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

pub fn convert_region_for_result<'db>(region: Region<'db>) -> crate::Lifetime {
    match region.kind() {
        rustc_type_ir::RegionKind::ReEarlyParam(early) => unimplemented!(),
        rustc_type_ir::RegionKind::ReBound(db, bound) => chalk_ir::Lifetime::new(
            Interner,
            chalk_ir::LifetimeData::BoundVar(chalk_ir::BoundVar::new(
                chalk_ir::DebruijnIndex::new(db.as_u32()),
                bound.var.as_usize(),
            )),
        ),
        rustc_type_ir::RegionKind::ReLateParam(_) => unimplemented!(),
        rustc_type_ir::RegionKind::ReStatic => {
            chalk_ir::Lifetime::new(Interner, chalk_ir::LifetimeData::Static)
        }
        rustc_type_ir::RegionKind::ReVar(vid) => chalk_ir::Lifetime::new(
            Interner,
            chalk_ir::LifetimeData::InferenceVar(chalk_ir::InferenceVar::from(vid.as_u32())),
        ),
        rustc_type_ir::RegionKind::RePlaceholder(placeholder) => chalk_ir::Lifetime::new(
            Interner,
            chalk_ir::LifetimeData::Placeholder(chalk_ir::PlaceholderIndex {
                idx: placeholder.bound.var.as_usize(),
                ui: chalk_ir::UniverseIndex { counter: placeholder.universe.as_usize() },
            }),
        ),
        rustc_type_ir::RegionKind::ReErased => {
            chalk_ir::Lifetime::new(Interner, chalk_ir::LifetimeData::Erased)
        }
        rustc_type_ir::RegionKind::ReError(_) => {
            chalk_ir::Lifetime::new(Interner, chalk_ir::LifetimeData::Error)
        }
    }
}
