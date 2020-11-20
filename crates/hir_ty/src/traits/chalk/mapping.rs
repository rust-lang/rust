//! This module contains the implementations of the `ToChalk` trait, which
//! handles conversion between our data types and their corresponding types in
//! Chalk (in both directions); plus some helper functions for more specialized
//! conversions.

use chalk_ir::{
    cast::Cast, fold::shift::Shift, interner::HasInterner, LifetimeData, PlaceholderIndex, Scalar,
    UniverseIndex,
};
use chalk_solve::rust_ir;

use base_db::salsa::InternKey;
use hir_def::{type_ref::Mutability, AssocContainerId, GenericDefId, Lookup, TypeAliasId};

use crate::{
    db::HirDatabase,
    primitive::{FloatBitness, FloatTy, IntBitness, IntTy, Signedness},
    traits::{Canonical, Obligation},
    ApplicationTy, CallableDefId, GenericPredicate, InEnvironment, OpaqueTy, OpaqueTyId,
    ProjectionPredicate, ProjectionTy, Substs, TraitEnvironment, TraitRef, Ty, TyKind, TypeCtor,
};

use super::interner::*;
use super::*;

impl ToChalk for Ty {
    type Chalk = chalk_ir::Ty<Interner>;
    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Ty<Interner> {
        match self {
            Ty::Apply(apply_ty) => match apply_ty.ctor {
                TypeCtor::Ref(m) => ref_to_chalk(db, m, apply_ty.parameters),
                TypeCtor::Array => array_to_chalk(db, apply_ty.parameters),
                TypeCtor::FnPtr { num_args: _, is_varargs } => {
                    let substitution =
                        chalk_ir::FnSubst(apply_ty.parameters.to_chalk(db).shifted_in(&Interner));
                    chalk_ir::TyKind::Function(chalk_ir::FnPointer {
                        num_binders: 0,
                        sig: chalk_ir::FnSig {
                            abi: (),
                            safety: chalk_ir::Safety::Safe,
                            variadic: is_varargs,
                        },
                        substitution,
                    })
                    .intern(&Interner)
                }
                TypeCtor::AssociatedType(type_alias) => {
                    let assoc_type = TypeAliasAsAssocType(type_alias);
                    let assoc_type_id = assoc_type.to_chalk(db);
                    let substitution = apply_ty.parameters.to_chalk(db);
                    chalk_ir::TyKind::AssociatedType(assoc_type_id, substitution).intern(&Interner)
                }

                TypeCtor::OpaqueType(impl_trait_id) => {
                    let id = impl_trait_id.to_chalk(db);
                    let substitution = apply_ty.parameters.to_chalk(db);
                    chalk_ir::TyKind::OpaqueType(id, substitution).intern(&Interner)
                }

                TypeCtor::ForeignType(type_alias) => {
                    let foreign_type = TypeAliasAsForeignType(type_alias);
                    let foreign_type_id = foreign_type.to_chalk(db);
                    chalk_ir::TyKind::Foreign(foreign_type_id).intern(&Interner)
                }

                TypeCtor::Bool => chalk_ir::TyKind::Scalar(Scalar::Bool).intern(&Interner),
                TypeCtor::Char => chalk_ir::TyKind::Scalar(Scalar::Char).intern(&Interner),
                TypeCtor::Int(int_ty) => {
                    chalk_ir::TyKind::Scalar(int_ty_to_chalk(int_ty)).intern(&Interner)
                }
                TypeCtor::Float(FloatTy { bitness: FloatBitness::X32 }) => {
                    chalk_ir::TyKind::Scalar(Scalar::Float(chalk_ir::FloatTy::F32))
                        .intern(&Interner)
                }
                TypeCtor::Float(FloatTy { bitness: FloatBitness::X64 }) => {
                    chalk_ir::TyKind::Scalar(Scalar::Float(chalk_ir::FloatTy::F64))
                        .intern(&Interner)
                }

                TypeCtor::Tuple { cardinality } => {
                    let substitution = apply_ty.parameters.to_chalk(db);
                    chalk_ir::TyKind::Tuple(cardinality.into(), substitution).intern(&Interner)
                }
                TypeCtor::RawPtr(mutability) => {
                    let ty = apply_ty.parameters[0].clone().to_chalk(db);
                    chalk_ir::TyKind::Raw(mutability.to_chalk(db), ty).intern(&Interner)
                }
                TypeCtor::Slice => {
                    chalk_ir::TyKind::Slice(apply_ty.parameters[0].clone().to_chalk(db))
                        .intern(&Interner)
                }
                TypeCtor::Str => chalk_ir::TyKind::Str.intern(&Interner),
                TypeCtor::FnDef(callable_def) => {
                    let id = callable_def.to_chalk(db);
                    let substitution = apply_ty.parameters.to_chalk(db);
                    chalk_ir::TyKind::FnDef(id, substitution).intern(&Interner)
                }
                TypeCtor::Never => chalk_ir::TyKind::Never.intern(&Interner),

                TypeCtor::Closure { def, expr } => {
                    let closure_id = db.intern_closure((def, expr));
                    let substitution = apply_ty.parameters.to_chalk(db);
                    chalk_ir::TyKind::Closure(closure_id.into(), substitution).intern(&Interner)
                }

                TypeCtor::Adt(adt_id) => {
                    let substitution = apply_ty.parameters.to_chalk(db);
                    chalk_ir::TyKind::Adt(chalk_ir::AdtId(adt_id), substitution).intern(&Interner)
                }
            },
            Ty::Projection(proj_ty) => {
                let associated_ty_id = TypeAliasAsAssocType(proj_ty.associated_ty).to_chalk(db);
                let substitution = proj_ty.parameters.to_chalk(db);
                chalk_ir::AliasTy::Projection(chalk_ir::ProjectionTy {
                    associated_ty_id,
                    substitution,
                })
                .cast(&Interner)
                .intern(&Interner)
            }
            Ty::Placeholder(id) => {
                let interned_id = db.intern_type_param_id(id);
                PlaceholderIndex {
                    ui: UniverseIndex::ROOT,
                    idx: interned_id.as_intern_id().as_usize(),
                }
                .to_ty::<Interner>(&Interner)
            }
            Ty::Bound(idx) => chalk_ir::TyKind::BoundVar(idx).intern(&Interner),
            Ty::Infer(_infer_ty) => panic!("uncanonicalized infer ty"),
            Ty::Dyn(predicates) => {
                let where_clauses = chalk_ir::QuantifiedWhereClauses::from_iter(
                    &Interner,
                    predicates.iter().filter(|p| !p.is_error()).cloned().map(|p| p.to_chalk(db)),
                );
                let bounded_ty = chalk_ir::DynTy {
                    bounds: make_binders(where_clauses, 1),
                    lifetime: LifetimeData::Static.intern(&Interner),
                };
                chalk_ir::TyKind::Dyn(bounded_ty).intern(&Interner)
            }
            Ty::Opaque(opaque_ty) => {
                let opaque_ty_id = opaque_ty.opaque_ty_id.to_chalk(db);
                let substitution = opaque_ty.parameters.to_chalk(db);
                chalk_ir::TyKind::Alias(chalk_ir::AliasTy::Opaque(chalk_ir::OpaqueTy {
                    opaque_ty_id,
                    substitution,
                }))
                .intern(&Interner)
            }
            Ty::Unknown => chalk_ir::TyKind::Error.intern(&Interner),
        }
    }
    fn from_chalk(db: &dyn HirDatabase, chalk: chalk_ir::Ty<Interner>) -> Self {
        match chalk.data(&Interner).kind.clone() {
            chalk_ir::TyKind::Error => Ty::Unknown,
            chalk_ir::TyKind::Array(ty, _size) => {
                Ty::apply(TypeCtor::Array, Substs::single(from_chalk(db, ty)))
            }
            chalk_ir::TyKind::Placeholder(idx) => {
                assert_eq!(idx.ui, UniverseIndex::ROOT);
                let interned_id = crate::db::GlobalTypeParamId::from_intern_id(
                    crate::salsa::InternId::from(idx.idx),
                );
                Ty::Placeholder(db.lookup_intern_type_param_id(interned_id))
            }
            chalk_ir::TyKind::Alias(chalk_ir::AliasTy::Projection(proj)) => {
                let associated_ty =
                    from_chalk::<TypeAliasAsAssocType, _>(db, proj.associated_ty_id).0;
                let parameters = from_chalk(db, proj.substitution);
                Ty::Projection(ProjectionTy { associated_ty, parameters })
            }
            chalk_ir::TyKind::Alias(chalk_ir::AliasTy::Opaque(opaque_ty)) => {
                let impl_trait_id = from_chalk(db, opaque_ty.opaque_ty_id);
                let parameters = from_chalk(db, opaque_ty.substitution);
                Ty::Opaque(OpaqueTy { opaque_ty_id: impl_trait_id, parameters })
            }
            chalk_ir::TyKind::Function(chalk_ir::FnPointer {
                num_binders,
                sig: chalk_ir::FnSig { variadic, .. },
                substitution,
                ..
            }) => {
                assert_eq!(num_binders, 0);
                let parameters: Substs = from_chalk(
                    db,
                    substitution.0.shifted_out(&Interner).expect("fn ptr should have no binders"),
                );
                Ty::Apply(ApplicationTy {
                    ctor: TypeCtor::FnPtr {
                        num_args: (parameters.len() - 1) as u16,
                        is_varargs: variadic,
                    },
                    parameters,
                })
            }
            chalk_ir::TyKind::BoundVar(idx) => Ty::Bound(idx),
            chalk_ir::TyKind::InferenceVar(_iv, _kind) => Ty::Unknown,
            chalk_ir::TyKind::Dyn(where_clauses) => {
                assert_eq!(where_clauses.bounds.binders.len(&Interner), 1);
                let predicates = where_clauses
                    .bounds
                    .skip_binders()
                    .iter(&Interner)
                    .map(|c| from_chalk(db, c.clone()))
                    .collect();
                Ty::Dyn(predicates)
            }

            chalk_ir::TyKind::Adt(struct_id, subst) => {
                apply_ty_from_chalk(db, TypeCtor::Adt(struct_id.0), subst)
            }
            chalk_ir::TyKind::AssociatedType(type_id, subst) => apply_ty_from_chalk(
                db,
                TypeCtor::AssociatedType(from_chalk::<TypeAliasAsAssocType, _>(db, type_id).0),
                subst,
            ),
            chalk_ir::TyKind::OpaqueType(opaque_type_id, subst) => {
                apply_ty_from_chalk(db, TypeCtor::OpaqueType(from_chalk(db, opaque_type_id)), subst)
            }

            chalk_ir::TyKind::Scalar(Scalar::Bool) => Ty::simple(TypeCtor::Bool),
            chalk_ir::TyKind::Scalar(Scalar::Char) => Ty::simple(TypeCtor::Char),
            chalk_ir::TyKind::Scalar(Scalar::Int(int_ty)) => Ty::simple(TypeCtor::Int(IntTy {
                signedness: Signedness::Signed,
                bitness: bitness_from_chalk_int(int_ty),
            })),
            chalk_ir::TyKind::Scalar(Scalar::Uint(uint_ty)) => Ty::simple(TypeCtor::Int(IntTy {
                signedness: Signedness::Unsigned,
                bitness: bitness_from_chalk_uint(uint_ty),
            })),
            chalk_ir::TyKind::Scalar(Scalar::Float(chalk_ir::FloatTy::F32)) => {
                Ty::simple(TypeCtor::Float(FloatTy { bitness: FloatBitness::X32 }))
            }
            chalk_ir::TyKind::Scalar(Scalar::Float(chalk_ir::FloatTy::F64)) => {
                Ty::simple(TypeCtor::Float(FloatTy { bitness: FloatBitness::X64 }))
            }
            chalk_ir::TyKind::Tuple(cardinality, subst) => {
                apply_ty_from_chalk(db, TypeCtor::Tuple { cardinality: cardinality as u16 }, subst)
            }
            chalk_ir::TyKind::Raw(mutability, ty) => {
                Ty::apply_one(TypeCtor::RawPtr(from_chalk(db, mutability)), from_chalk(db, ty))
            }
            chalk_ir::TyKind::Slice(ty) => Ty::apply_one(TypeCtor::Slice, from_chalk(db, ty)),
            chalk_ir::TyKind::Ref(mutability, _lifetime, ty) => {
                Ty::apply_one(TypeCtor::Ref(from_chalk(db, mutability)), from_chalk(db, ty))
            }
            chalk_ir::TyKind::Str => Ty::simple(TypeCtor::Str),
            chalk_ir::TyKind::Never => Ty::simple(TypeCtor::Never),

            chalk_ir::TyKind::FnDef(fn_def_id, subst) => {
                let callable_def = from_chalk(db, fn_def_id);
                apply_ty_from_chalk(db, TypeCtor::FnDef(callable_def), subst)
            }

            chalk_ir::TyKind::Closure(id, subst) => {
                let id: crate::db::ClosureId = id.into();
                let (def, expr) = db.lookup_intern_closure(id);
                apply_ty_from_chalk(db, TypeCtor::Closure { def, expr }, subst)
            }

            chalk_ir::TyKind::Foreign(foreign_def_id) => Ty::simple(TypeCtor::ForeignType(
                from_chalk::<TypeAliasAsForeignType, _>(db, foreign_def_id).0,
            )),
            chalk_ir::TyKind::Generator(_, _) => unimplemented!(), // FIXME
            chalk_ir::TyKind::GeneratorWitness(_, _) => unimplemented!(), // FIXME
        }
    }
}

fn apply_ty_from_chalk(
    db: &dyn HirDatabase,
    ctor: TypeCtor,
    subst: chalk_ir::Substitution<Interner>,
) -> Ty {
    Ty::Apply(ApplicationTy { ctor, parameters: from_chalk(db, subst) })
}

/// We currently don't model lifetimes, but Chalk does. So, we have to insert a
/// fake lifetime here, because Chalks built-in logic may expect it to be there.
fn ref_to_chalk(
    db: &dyn HirDatabase,
    mutability: Mutability,
    subst: Substs,
) -> chalk_ir::Ty<Interner> {
    let arg = subst[0].clone().to_chalk(db);
    let lifetime = LifetimeData::Static.intern(&Interner);
    chalk_ir::TyKind::Ref(mutability.to_chalk(db), lifetime, arg).intern(&Interner)
}

/// We currently don't model constants, but Chalk does. So, we have to insert a
/// fake constant here, because Chalks built-in logic may expect it to be there.
fn array_to_chalk(db: &dyn HirDatabase, subst: Substs) -> chalk_ir::Ty<Interner> {
    let arg = subst[0].clone().to_chalk(db);
    let usize_ty =
        chalk_ir::TyKind::Scalar(Scalar::Uint(chalk_ir::UintTy::Usize)).intern(&Interner);
    let const_ = chalk_ir::ConstData {
        ty: usize_ty,
        value: chalk_ir::ConstValue::Concrete(chalk_ir::ConcreteConst { interned: () }),
    }
    .intern(&Interner);
    chalk_ir::TyKind::Array(arg, const_).intern(&Interner)
}

impl ToChalk for Substs {
    type Chalk = chalk_ir::Substitution<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Substitution<Interner> {
        chalk_ir::Substitution::from_iter(&Interner, self.iter().map(|ty| ty.clone().to_chalk(db)))
    }

    fn from_chalk(db: &dyn HirDatabase, parameters: chalk_ir::Substitution<Interner>) -> Substs {
        let tys = parameters
            .iter(&Interner)
            .map(|p| match p.ty(&Interner) {
                Some(ty) => from_chalk(db, ty.clone()),
                None => unimplemented!(),
            })
            .collect();
        Substs(tys)
    }
}

impl ToChalk for TraitRef {
    type Chalk = chalk_ir::TraitRef<Interner>;

    fn to_chalk(self: TraitRef, db: &dyn HirDatabase) -> chalk_ir::TraitRef<Interner> {
        let trait_id = self.trait_.to_chalk(db);
        let substitution = self.substs.to_chalk(db);
        chalk_ir::TraitRef { trait_id, substitution }
    }

    fn from_chalk(db: &dyn HirDatabase, trait_ref: chalk_ir::TraitRef<Interner>) -> Self {
        let trait_ = from_chalk(db, trait_ref.trait_id);
        let substs = from_chalk(db, trait_ref.substitution);
        TraitRef { trait_, substs }
    }
}

impl ToChalk for hir_def::TraitId {
    type Chalk = TraitId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> TraitId {
        chalk_ir::TraitId(self.as_intern_id())
    }

    fn from_chalk(_db: &dyn HirDatabase, trait_id: TraitId) -> hir_def::TraitId {
        InternKey::from_intern_id(trait_id.0)
    }
}

impl ToChalk for OpaqueTyId {
    type Chalk = chalk_ir::OpaqueTyId<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::OpaqueTyId<Interner> {
        db.intern_impl_trait_id(self).into()
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        opaque_ty_id: chalk_ir::OpaqueTyId<Interner>,
    ) -> OpaqueTyId {
        db.lookup_intern_impl_trait_id(opaque_ty_id.into())
    }
}

fn bitness_from_chalk_uint(uint_ty: chalk_ir::UintTy) -> IntBitness {
    use chalk_ir::UintTy;

    match uint_ty {
        UintTy::Usize => IntBitness::Xsize,
        UintTy::U8 => IntBitness::X8,
        UintTy::U16 => IntBitness::X16,
        UintTy::U32 => IntBitness::X32,
        UintTy::U64 => IntBitness::X64,
        UintTy::U128 => IntBitness::X128,
    }
}

fn bitness_from_chalk_int(int_ty: chalk_ir::IntTy) -> IntBitness {
    use chalk_ir::IntTy;

    match int_ty {
        IntTy::Isize => IntBitness::Xsize,
        IntTy::I8 => IntBitness::X8,
        IntTy::I16 => IntBitness::X16,
        IntTy::I32 => IntBitness::X32,
        IntTy::I64 => IntBitness::X64,
        IntTy::I128 => IntBitness::X128,
    }
}

fn int_ty_to_chalk(int_ty: IntTy) -> Scalar {
    use chalk_ir::{IntTy, UintTy};

    match int_ty.signedness {
        Signedness::Signed => Scalar::Int(match int_ty.bitness {
            IntBitness::Xsize => IntTy::Isize,
            IntBitness::X8 => IntTy::I8,
            IntBitness::X16 => IntTy::I16,
            IntBitness::X32 => IntTy::I32,
            IntBitness::X64 => IntTy::I64,
            IntBitness::X128 => IntTy::I128,
        }),
        Signedness::Unsigned => Scalar::Uint(match int_ty.bitness {
            IntBitness::Xsize => UintTy::Usize,
            IntBitness::X8 => UintTy::U8,
            IntBitness::X16 => UintTy::U16,
            IntBitness::X32 => UintTy::U32,
            IntBitness::X64 => UintTy::U64,
            IntBitness::X128 => UintTy::U128,
        }),
    }
}

impl ToChalk for Mutability {
    type Chalk = chalk_ir::Mutability;
    fn to_chalk(self, _db: &dyn HirDatabase) -> Self::Chalk {
        match self {
            Mutability::Shared => chalk_ir::Mutability::Not,
            Mutability::Mut => chalk_ir::Mutability::Mut,
        }
    }
    fn from_chalk(_db: &dyn HirDatabase, chalk: Self::Chalk) -> Self {
        match chalk {
            chalk_ir::Mutability::Mut => Mutability::Mut,
            chalk_ir::Mutability::Not => Mutability::Shared,
        }
    }
}

impl ToChalk for hir_def::ImplId {
    type Chalk = ImplId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> ImplId {
        chalk_ir::ImplId(self.as_intern_id())
    }

    fn from_chalk(_db: &dyn HirDatabase, impl_id: ImplId) -> hir_def::ImplId {
        InternKey::from_intern_id(impl_id.0)
    }
}

impl ToChalk for hir_def::AdtId {
    type Chalk = AdtId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> Self::Chalk {
        chalk_ir::AdtId(self.into())
    }

    fn from_chalk(_db: &dyn HirDatabase, id: AdtId) -> Self {
        id.0
    }
}

impl ToChalk for CallableDefId {
    type Chalk = FnDefId;

    fn to_chalk(self, db: &dyn HirDatabase) -> FnDefId {
        db.intern_callable_def(self).into()
    }

    fn from_chalk(db: &dyn HirDatabase, fn_def_id: FnDefId) -> CallableDefId {
        db.lookup_intern_callable_def(fn_def_id.into())
    }
}

pub(crate) struct TypeAliasAsAssocType(pub(crate) TypeAliasId);

impl ToChalk for TypeAliasAsAssocType {
    type Chalk = AssocTypeId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> AssocTypeId {
        chalk_ir::AssocTypeId(self.0.as_intern_id())
    }

    fn from_chalk(_db: &dyn HirDatabase, assoc_type_id: AssocTypeId) -> TypeAliasAsAssocType {
        TypeAliasAsAssocType(InternKey::from_intern_id(assoc_type_id.0))
    }
}

pub(crate) struct TypeAliasAsForeignType(pub(crate) TypeAliasId);

impl ToChalk for TypeAliasAsForeignType {
    type Chalk = ForeignDefId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> ForeignDefId {
        chalk_ir::ForeignDefId(self.0.as_intern_id())
    }

    fn from_chalk(_db: &dyn HirDatabase, foreign_def_id: ForeignDefId) -> TypeAliasAsForeignType {
        TypeAliasAsForeignType(InternKey::from_intern_id(foreign_def_id.0))
    }
}

pub(crate) struct TypeAliasAsValue(pub(crate) TypeAliasId);

impl ToChalk for TypeAliasAsValue {
    type Chalk = AssociatedTyValueId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> AssociatedTyValueId {
        rust_ir::AssociatedTyValueId(self.0.as_intern_id())
    }

    fn from_chalk(
        _db: &dyn HirDatabase,
        assoc_ty_value_id: AssociatedTyValueId,
    ) -> TypeAliasAsValue {
        TypeAliasAsValue(TypeAliasId::from_intern_id(assoc_ty_value_id.0))
    }
}

impl ToChalk for GenericPredicate {
    type Chalk = chalk_ir::QuantifiedWhereClause<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::QuantifiedWhereClause<Interner> {
        match self {
            GenericPredicate::Implemented(trait_ref) => {
                let chalk_trait_ref = trait_ref.to_chalk(db);
                let chalk_trait_ref = chalk_trait_ref.shifted_in(&Interner);
                make_binders(chalk_ir::WhereClause::Implemented(chalk_trait_ref), 0)
            }
            GenericPredicate::Projection(projection_pred) => {
                let ty = projection_pred.ty.to_chalk(db).shifted_in(&Interner);
                let projection = projection_pred.projection_ty.to_chalk(db).shifted_in(&Interner);
                let alias = chalk_ir::AliasTy::Projection(projection);
                make_binders(chalk_ir::WhereClause::AliasEq(chalk_ir::AliasEq { alias, ty }), 0)
            }
            GenericPredicate::Error => panic!("tried passing GenericPredicate::Error to Chalk"),
        }
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        where_clause: chalk_ir::QuantifiedWhereClause<Interner>,
    ) -> GenericPredicate {
        // we don't produce any where clauses with binders and can't currently deal with them
        match where_clause
            .skip_binders()
            .clone()
            .shifted_out(&Interner)
            .expect("unexpected bound vars in where clause")
        {
            chalk_ir::WhereClause::Implemented(tr) => {
                GenericPredicate::Implemented(from_chalk(db, tr))
            }
            chalk_ir::WhereClause::AliasEq(projection_eq) => {
                let projection_ty = from_chalk(
                    db,
                    match projection_eq.alias {
                        chalk_ir::AliasTy::Projection(p) => p,
                        _ => unimplemented!(),
                    },
                );
                let ty = from_chalk(db, projection_eq.ty);
                GenericPredicate::Projection(ProjectionPredicate { projection_ty, ty })
            }

            chalk_ir::WhereClause::LifetimeOutlives(_) => {
                // we shouldn't get these from Chalk
                panic!("encountered LifetimeOutlives from Chalk")
            }

            chalk_ir::WhereClause::TypeOutlives(_) => {
                // we shouldn't get these from Chalk
                panic!("encountered TypeOutlives from Chalk")
            }
        }
    }
}

impl ToChalk for ProjectionTy {
    type Chalk = chalk_ir::ProjectionTy<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::ProjectionTy<Interner> {
        chalk_ir::ProjectionTy {
            associated_ty_id: TypeAliasAsAssocType(self.associated_ty).to_chalk(db),
            substitution: self.parameters.to_chalk(db),
        }
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        projection_ty: chalk_ir::ProjectionTy<Interner>,
    ) -> ProjectionTy {
        ProjectionTy {
            associated_ty: from_chalk::<TypeAliasAsAssocType, _>(
                db,
                projection_ty.associated_ty_id,
            )
            .0,
            parameters: from_chalk(db, projection_ty.substitution),
        }
    }
}

impl ToChalk for ProjectionPredicate {
    type Chalk = chalk_ir::AliasEq<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::AliasEq<Interner> {
        chalk_ir::AliasEq {
            alias: chalk_ir::AliasTy::Projection(self.projection_ty.to_chalk(db)),
            ty: self.ty.to_chalk(db),
        }
    }

    fn from_chalk(_db: &dyn HirDatabase, _normalize: chalk_ir::AliasEq<Interner>) -> Self {
        unimplemented!()
    }
}

impl ToChalk for Obligation {
    type Chalk = chalk_ir::DomainGoal<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::DomainGoal<Interner> {
        match self {
            Obligation::Trait(tr) => tr.to_chalk(db).cast(&Interner),
            Obligation::Projection(pr) => pr.to_chalk(db).cast(&Interner),
        }
    }

    fn from_chalk(_db: &dyn HirDatabase, _goal: chalk_ir::DomainGoal<Interner>) -> Self {
        unimplemented!()
    }
}

impl<T> ToChalk for Canonical<T>
where
    T: ToChalk,
    T::Chalk: HasInterner<Interner = Interner>,
{
    type Chalk = chalk_ir::Canonical<T::Chalk>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Canonical<T::Chalk> {
        let kinds = self
            .kinds
            .iter()
            .map(|k| match k {
                TyKind::General => chalk_ir::TyVariableKind::General,
                TyKind::Integer => chalk_ir::TyVariableKind::Integer,
                TyKind::Float => chalk_ir::TyVariableKind::Float,
            })
            .map(|tk| {
                chalk_ir::CanonicalVarKind::new(
                    chalk_ir::VariableKind::Ty(tk),
                    chalk_ir::UniverseIndex::ROOT,
                )
            });
        let value = self.value.to_chalk(db);
        chalk_ir::Canonical {
            value,
            binders: chalk_ir::CanonicalVarKinds::from_iter(&Interner, kinds),
        }
    }

    fn from_chalk(db: &dyn HirDatabase, canonical: chalk_ir::Canonical<T::Chalk>) -> Canonical<T> {
        let kinds = canonical
            .binders
            .iter(&Interner)
            .map(|k| match k.kind {
                chalk_ir::VariableKind::Ty(tk) => match tk {
                    chalk_ir::TyVariableKind::General => TyKind::General,
                    chalk_ir::TyVariableKind::Integer => TyKind::Integer,
                    chalk_ir::TyVariableKind::Float => TyKind::Float,
                },
                // HACK: Chalk can sometimes return new lifetime variables. We
                // want to just skip them, but to not mess up the indices of
                // other variables, we'll just create a new type variable in
                // their place instead. This should not matter (we never see the
                // actual *uses* of the lifetime variable).
                chalk_ir::VariableKind::Lifetime => TyKind::General,
                chalk_ir::VariableKind::Const(_) => panic!("unexpected const from Chalk"),
            })
            .collect();
        Canonical { kinds, value: from_chalk(db, canonical.value) }
    }
}

impl ToChalk for Arc<TraitEnvironment> {
    type Chalk = chalk_ir::Environment<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Environment<Interner> {
        let mut clauses = Vec::new();
        for pred in &self.predicates {
            if pred.is_error() {
                // for env, we just ignore errors
                continue;
            }
            let program_clause: chalk_ir::ProgramClause<Interner> =
                pred.clone().to_chalk(db).cast(&Interner);
            clauses.push(program_clause.into_from_env_clause(&Interner));
        }
        chalk_ir::Environment::new(&Interner).add_clauses(&Interner, clauses)
    }

    fn from_chalk(
        _db: &dyn HirDatabase,
        _env: chalk_ir::Environment<Interner>,
    ) -> Arc<TraitEnvironment> {
        unimplemented!()
    }
}

impl<T: ToChalk> ToChalk for InEnvironment<T>
where
    T::Chalk: chalk_ir::interner::HasInterner<Interner = Interner>,
{
    type Chalk = chalk_ir::InEnvironment<T::Chalk>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::InEnvironment<T::Chalk> {
        chalk_ir::InEnvironment {
            environment: self.environment.to_chalk(db),
            goal: self.value.to_chalk(db),
        }
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        in_env: chalk_ir::InEnvironment<T::Chalk>,
    ) -> InEnvironment<T> {
        InEnvironment {
            environment: from_chalk(db, in_env.environment),
            value: from_chalk(db, in_env.goal),
        }
    }
}

pub(super) fn make_binders<T>(value: T, num_vars: usize) -> chalk_ir::Binders<T>
where
    T: HasInterner<Interner = Interner>,
{
    chalk_ir::Binders::new(
        chalk_ir::VariableKinds::from_iter(
            &Interner,
            std::iter::repeat(chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General))
                .take(num_vars),
        ),
        value,
    )
}

pub(super) fn convert_where_clauses(
    db: &dyn HirDatabase,
    def: GenericDefId,
    substs: &Substs,
) -> Vec<chalk_ir::QuantifiedWhereClause<Interner>> {
    let generic_predicates = db.generic_predicates(def);
    let mut result = Vec::with_capacity(generic_predicates.len());
    for pred in generic_predicates.iter() {
        if pred.value.is_error() {
            // skip errored predicates completely
            continue;
        }
        result.push(pred.clone().subst(substs).to_chalk(db));
    }
    result
}

pub(super) fn generic_predicate_to_inline_bound(
    db: &dyn HirDatabase,
    pred: &GenericPredicate,
    self_ty: &Ty,
) -> Option<rust_ir::InlineBound<Interner>> {
    // An InlineBound is like a GenericPredicate, except the self type is left out.
    // We don't have a special type for this, but Chalk does.
    match pred {
        GenericPredicate::Implemented(trait_ref) => {
            if &trait_ref.substs[0] != self_ty {
                // we can only convert predicates back to type bounds if they
                // have the expected self type
                return None;
            }
            let args_no_self = trait_ref.substs[1..]
                .iter()
                .map(|ty| ty.clone().to_chalk(db).cast(&Interner))
                .collect();
            let trait_bound =
                rust_ir::TraitBound { trait_id: trait_ref.trait_.to_chalk(db), args_no_self };
            Some(rust_ir::InlineBound::TraitBound(trait_bound))
        }
        GenericPredicate::Projection(proj) => {
            if &proj.projection_ty.parameters[0] != self_ty {
                return None;
            }
            let trait_ = match proj.projection_ty.associated_ty.lookup(db.upcast()).container {
                AssocContainerId::TraitId(t) => t,
                _ => panic!("associated type not in trait"),
            };
            let args_no_self = proj.projection_ty.parameters[1..]
                .iter()
                .map(|ty| ty.clone().to_chalk(db).cast(&Interner))
                .collect();
            let alias_eq_bound = rust_ir::AliasEqBound {
                value: proj.ty.clone().to_chalk(db),
                trait_bound: rust_ir::TraitBound { trait_id: trait_.to_chalk(db), args_no_self },
                associated_ty_id: TypeAliasAsAssocType(proj.projection_ty.associated_ty)
                    .to_chalk(db),
                parameters: Vec::new(), // FIXME we don't support generic associated types yet
            };
            Some(rust_ir::InlineBound::AliasEqBound(alias_eq_bound))
        }
        GenericPredicate::Error => None,
    }
}
