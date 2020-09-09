//! This module contains the implementations of the `ToChalk` trait, which
//! handles conversion between our data types and their corresponding types in
//! Chalk (in both directions); plus some helper functions for more specialized
//! conversions.

use chalk_ir::{
    cast::Cast, fold::shift::Shift, interner::HasInterner, PlaceholderIndex, Scalar, TypeName,
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
                    let substitution = apply_ty.parameters.to_chalk(db).shifted_in(&Interner);
                    chalk_ir::TyData::Function(chalk_ir::FnPointer {
                        num_binders: 0,
                        abi: (),
                        safety: chalk_ir::Safety::Safe,
                        variadic: is_varargs,
                        substitution,
                    })
                    .intern(&Interner)
                }
                _ => {
                    let name = apply_ty.ctor.to_chalk(db);
                    let substitution = apply_ty.parameters.to_chalk(db);
                    chalk_ir::ApplicationTy { name, substitution }.cast(&Interner).intern(&Interner)
                }
            },
            Ty::Projection(proj_ty) => {
                let associated_ty_id = proj_ty.associated_ty.to_chalk(db);
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
            Ty::Bound(idx) => chalk_ir::TyData::BoundVar(idx).intern(&Interner),
            Ty::Infer(_infer_ty) => panic!("uncanonicalized infer ty"),
            Ty::Dyn(predicates) => {
                let where_clauses = chalk_ir::QuantifiedWhereClauses::from_iter(
                    &Interner,
                    predicates.iter().filter(|p| !p.is_error()).cloned().map(|p| p.to_chalk(db)),
                );
                let bounded_ty = chalk_ir::DynTy {
                    bounds: make_binders(where_clauses, 1),
                    lifetime: FAKE_PLACEHOLDER.to_lifetime(&Interner),
                };
                chalk_ir::TyData::Dyn(bounded_ty).intern(&Interner)
            }
            Ty::Opaque(opaque_ty) => {
                let opaque_ty_id = opaque_ty.opaque_ty_id.to_chalk(db);
                let substitution = opaque_ty.parameters.to_chalk(db);
                chalk_ir::TyData::Alias(chalk_ir::AliasTy::Opaque(chalk_ir::OpaqueTy {
                    opaque_ty_id,
                    substitution,
                }))
                .intern(&Interner)
            }
            Ty::Unknown => {
                let substitution = chalk_ir::Substitution::empty(&Interner);
                let name = TypeName::Error;
                chalk_ir::ApplicationTy { name, substitution }.cast(&Interner).intern(&Interner)
            }
        }
    }
    fn from_chalk(db: &dyn HirDatabase, chalk: chalk_ir::Ty<Interner>) -> Self {
        match chalk.data(&Interner).clone() {
            chalk_ir::TyData::Apply(apply_ty) => match apply_ty.name {
                TypeName::Error => Ty::Unknown,
                TypeName::Ref(m) => ref_from_chalk(db, m, apply_ty.substitution),
                TypeName::Array => array_from_chalk(db, apply_ty.substitution),
                _ => {
                    let ctor = from_chalk(db, apply_ty.name);
                    let parameters = from_chalk(db, apply_ty.substitution);
                    Ty::Apply(ApplicationTy { ctor, parameters })
                }
            },
            chalk_ir::TyData::Placeholder(idx) => {
                assert_eq!(idx.ui, UniverseIndex::ROOT);
                let interned_id = crate::db::GlobalTypeParamId::from_intern_id(
                    crate::salsa::InternId::from(idx.idx),
                );
                Ty::Placeholder(db.lookup_intern_type_param_id(interned_id))
            }
            chalk_ir::TyData::Alias(chalk_ir::AliasTy::Projection(proj)) => {
                let associated_ty = from_chalk(db, proj.associated_ty_id);
                let parameters = from_chalk(db, proj.substitution);
                Ty::Projection(ProjectionTy { associated_ty, parameters })
            }
            chalk_ir::TyData::Alias(chalk_ir::AliasTy::Opaque(opaque_ty)) => {
                let impl_trait_id = from_chalk(db, opaque_ty.opaque_ty_id);
                let parameters = from_chalk(db, opaque_ty.substitution);
                Ty::Opaque(OpaqueTy { opaque_ty_id: impl_trait_id, parameters })
            }
            chalk_ir::TyData::Function(chalk_ir::FnPointer {
                num_binders,
                variadic,
                substitution,
                ..
            }) => {
                assert_eq!(num_binders, 0);
                let parameters: Substs = from_chalk(
                    db,
                    substitution.shifted_out(&Interner).expect("fn ptr should have no binders"),
                );
                Ty::Apply(ApplicationTy {
                    ctor: TypeCtor::FnPtr {
                        num_args: (parameters.len() - 1) as u16,
                        is_varargs: variadic,
                    },
                    parameters,
                })
            }
            chalk_ir::TyData::BoundVar(idx) => Ty::Bound(idx),
            chalk_ir::TyData::InferenceVar(_iv, _kind) => Ty::Unknown,
            chalk_ir::TyData::Dyn(where_clauses) => {
                assert_eq!(where_clauses.bounds.binders.len(&Interner), 1);
                let predicates = where_clauses
                    .bounds
                    .skip_binders()
                    .iter(&Interner)
                    .map(|c| from_chalk(db, c.clone()))
                    .collect();
                Ty::Dyn(predicates)
            }
        }
    }
}

const FAKE_PLACEHOLDER: PlaceholderIndex =
    PlaceholderIndex { ui: UniverseIndex::ROOT, idx: usize::MAX };

/// We currently don't model lifetimes, but Chalk does. So, we have to insert a
/// fake lifetime here, because Chalks built-in logic may expect it to be there.
fn ref_to_chalk(
    db: &dyn HirDatabase,
    mutability: Mutability,
    subst: Substs,
) -> chalk_ir::Ty<Interner> {
    let arg = subst[0].clone().to_chalk(db);
    let lifetime = FAKE_PLACEHOLDER.to_lifetime(&Interner);
    chalk_ir::ApplicationTy {
        name: TypeName::Ref(mutability.to_chalk(db)),
        substitution: chalk_ir::Substitution::from_iter(
            &Interner,
            vec![lifetime.cast(&Interner), arg.cast(&Interner)],
        ),
    }
    .intern(&Interner)
}

/// Here we remove the lifetime from the type we got from Chalk.
fn ref_from_chalk(
    db: &dyn HirDatabase,
    mutability: chalk_ir::Mutability,
    subst: chalk_ir::Substitution<Interner>,
) -> Ty {
    let tys = subst
        .iter(&Interner)
        .filter_map(|p| Some(from_chalk(db, p.ty(&Interner)?.clone())))
        .collect();
    Ty::apply(TypeCtor::Ref(from_chalk(db, mutability)), Substs(tys))
}

/// We currently don't model constants, but Chalk does. So, we have to insert a
/// fake constant here, because Chalks built-in logic may expect it to be there.
fn array_to_chalk(db: &dyn HirDatabase, subst: Substs) -> chalk_ir::Ty<Interner> {
    let arg = subst[0].clone().to_chalk(db);
    let usize_ty = chalk_ir::ApplicationTy {
        name: TypeName::Scalar(Scalar::Uint(chalk_ir::UintTy::Usize)),
        substitution: chalk_ir::Substitution::empty(&Interner),
    }
    .intern(&Interner);
    let const_ = FAKE_PLACEHOLDER.to_const(&Interner, usize_ty);
    chalk_ir::ApplicationTy {
        name: TypeName::Array,
        substitution: chalk_ir::Substitution::from_iter(
            &Interner,
            vec![arg.cast(&Interner), const_.cast(&Interner)],
        ),
    }
    .intern(&Interner)
}

/// Here we remove the const from the type we got from Chalk.
fn array_from_chalk(db: &dyn HirDatabase, subst: chalk_ir::Substitution<Interner>) -> Ty {
    let tys = subst
        .iter(&Interner)
        .filter_map(|p| Some(from_chalk(db, p.ty(&Interner)?.clone())))
        .collect();
    Ty::apply(TypeCtor::Array, Substs(tys))
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

impl ToChalk for TypeCtor {
    type Chalk = TypeName<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> TypeName<Interner> {
        match self {
            TypeCtor::AssociatedType(type_alias) => {
                let type_id = type_alias.to_chalk(db);
                TypeName::AssociatedType(type_id)
            }

            TypeCtor::OpaqueType(impl_trait_id) => {
                let id = impl_trait_id.to_chalk(db);
                TypeName::OpaqueType(id)
            }

            TypeCtor::Bool => TypeName::Scalar(Scalar::Bool),
            TypeCtor::Char => TypeName::Scalar(Scalar::Char),
            TypeCtor::Int(int_ty) => TypeName::Scalar(int_ty_to_chalk(int_ty)),
            TypeCtor::Float(FloatTy { bitness: FloatBitness::X32 }) => {
                TypeName::Scalar(Scalar::Float(chalk_ir::FloatTy::F32))
            }
            TypeCtor::Float(FloatTy { bitness: FloatBitness::X64 }) => {
                TypeName::Scalar(Scalar::Float(chalk_ir::FloatTy::F64))
            }

            TypeCtor::Tuple { cardinality } => TypeName::Tuple(cardinality.into()),
            TypeCtor::RawPtr(mutability) => TypeName::Raw(mutability.to_chalk(db)),
            TypeCtor::Slice => TypeName::Slice,
            TypeCtor::Array => TypeName::Array,
            TypeCtor::Ref(mutability) => TypeName::Ref(mutability.to_chalk(db)),
            TypeCtor::Str => TypeName::Str,
            TypeCtor::FnDef(callable_def) => {
                let id = callable_def.to_chalk(db);
                TypeName::FnDef(id)
            }
            TypeCtor::Never => TypeName::Never,

            TypeCtor::Closure { def, expr } => {
                let closure_id = db.intern_closure((def, expr));
                TypeName::Closure(closure_id.into())
            }

            TypeCtor::Adt(adt_id) => TypeName::Adt(chalk_ir::AdtId(adt_id)),

            TypeCtor::FnPtr { .. } => {
                // This should not be reached, since Chalk doesn't represent
                // function pointers with TypeName
                unreachable!()
            }
        }
    }

    fn from_chalk(db: &dyn HirDatabase, type_name: TypeName<Interner>) -> TypeCtor {
        match type_name {
            TypeName::Adt(struct_id) => TypeCtor::Adt(struct_id.0),
            TypeName::AssociatedType(type_id) => TypeCtor::AssociatedType(from_chalk(db, type_id)),
            TypeName::OpaqueType(opaque_type_id) => {
                TypeCtor::OpaqueType(from_chalk(db, opaque_type_id))
            }

            TypeName::Scalar(Scalar::Bool) => TypeCtor::Bool,
            TypeName::Scalar(Scalar::Char) => TypeCtor::Char,
            TypeName::Scalar(Scalar::Int(int_ty)) => TypeCtor::Int(IntTy {
                signedness: Signedness::Signed,
                bitness: bitness_from_chalk_int(int_ty),
            }),
            TypeName::Scalar(Scalar::Uint(uint_ty)) => TypeCtor::Int(IntTy {
                signedness: Signedness::Unsigned,
                bitness: bitness_from_chalk_uint(uint_ty),
            }),
            TypeName::Scalar(Scalar::Float(chalk_ir::FloatTy::F32)) => {
                TypeCtor::Float(FloatTy { bitness: FloatBitness::X32 })
            }
            TypeName::Scalar(Scalar::Float(chalk_ir::FloatTy::F64)) => {
                TypeCtor::Float(FloatTy { bitness: FloatBitness::X64 })
            }
            TypeName::Tuple(cardinality) => TypeCtor::Tuple { cardinality: cardinality as u16 },
            TypeName::Raw(mutability) => TypeCtor::RawPtr(from_chalk(db, mutability)),
            TypeName::Slice => TypeCtor::Slice,
            TypeName::Ref(mutability) => TypeCtor::Ref(from_chalk(db, mutability)),
            TypeName::Str => TypeCtor::Str,
            TypeName::Never => TypeCtor::Never,

            TypeName::FnDef(fn_def_id) => {
                let callable_def = from_chalk(db, fn_def_id);
                TypeCtor::FnDef(callable_def)
            }
            TypeName::Array => TypeCtor::Array,

            TypeName::Closure(id) => {
                let id: crate::db::ClosureId = id.into();
                let (def, expr) = db.lookup_intern_closure(id);
                TypeCtor::Closure { def, expr }
            }

            TypeName::Error => {
                // this should not be reached, since we don't represent TypeName::Error with TypeCtor
                unreachable!()
            }
        }
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

impl ToChalk for TypeAliasId {
    type Chalk = AssocTypeId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> AssocTypeId {
        chalk_ir::AssocTypeId(self.as_intern_id())
    }

    fn from_chalk(_db: &dyn HirDatabase, type_alias_id: AssocTypeId) -> TypeAliasId {
        InternKey::from_intern_id(type_alias_id.0)
    }
}

pub struct TypeAliasAsValue(pub TypeAliasId);

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
            associated_ty_id: self.associated_ty.to_chalk(db),
            substitution: self.parameters.to_chalk(db),
        }
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        projection_ty: chalk_ir::ProjectionTy<Interner>,
    ) -> ProjectionTy {
        ProjectionTy {
            associated_ty: from_chalk(db, projection_ty.associated_ty_id),
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
                TyKind::General => chalk_ir::TyKind::General,
                TyKind::Integer => chalk_ir::TyKind::Integer,
                TyKind::Float => chalk_ir::TyKind::Float,
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
                    chalk_ir::TyKind::General => TyKind::General,
                    chalk_ir::TyKind::Integer => TyKind::Integer,
                    chalk_ir::TyKind::Float => TyKind::Float,
                },
                chalk_ir::VariableKind::Lifetime => panic!("unexpected lifetime from Chalk"),
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
            std::iter::repeat(chalk_ir::VariableKind::Ty(chalk_ir::TyKind::General)).take(num_vars),
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
                associated_ty_id: proj.projection_ty.associated_ty.to_chalk(db),
                parameters: Vec::new(), // FIXME we don't support generic associated types yet
            };
            Some(rust_ir::InlineBound::AliasEqBound(alias_eq_bound))
        }
        GenericPredicate::Error => None,
    }
}
