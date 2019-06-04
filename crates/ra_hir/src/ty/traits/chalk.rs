//! Conversion code from/to Chalk.
use std::sync::Arc;

use log::debug;

use chalk_ir::{TypeId, ImplId, TypeKindId, ProjectionTy, Parameter, Identifier, cast::Cast, PlaceholderIndex, UniverseIndex, TypeName};
use chalk_rust_ir::{AssociatedTyDatum, TraitDatum, StructDatum, ImplDatum};

use test_utils::tested_by;
use ra_db::salsa::{InternId, InternKey};

use crate::{
    Trait, HasGenericParams, ImplBlock,
    db::HirDatabase,
    ty::{TraitRef, Ty, ApplicationTy, TypeCtor, Substs, GenericPredicate, CallableDef},
    ty::display::HirDisplay,
    generics::GenericDef,
};
use super::ChalkContext;

/// This represents a trait whose name we could not resolve.
const UNKNOWN_TRAIT: chalk_ir::TraitId =
    chalk_ir::TraitId(chalk_ir::RawId { index: u32::max_value() });

pub(super) trait ToChalk {
    type Chalk;
    fn to_chalk(self, db: &impl HirDatabase) -> Self::Chalk;
    fn from_chalk(db: &impl HirDatabase, chalk: Self::Chalk) -> Self;
}

pub(super) fn from_chalk<T, ChalkT>(db: &impl HirDatabase, chalk: ChalkT) -> T
where
    T: ToChalk<Chalk = ChalkT>,
{
    T::from_chalk(db, chalk)
}

impl ToChalk for Ty {
    type Chalk = chalk_ir::Ty;
    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::Ty {
        match self {
            Ty::Apply(apply_ty) => {
                let struct_id = apply_ty.ctor.to_chalk(db);
                let name = TypeName::TypeKindId(struct_id.into());
                let parameters = apply_ty.parameters.to_chalk(db);
                chalk_ir::ApplicationTy { name, parameters }.cast()
            }
            Ty::Param { idx, .. } => {
                PlaceholderIndex { ui: UniverseIndex::ROOT, idx: idx as usize }.to_ty()
            }
            Ty::Bound(idx) => chalk_ir::Ty::BoundVar(idx as usize),
            Ty::Infer(_infer_ty) => panic!("uncanonicalized infer ty"),
            // FIXME this is clearly incorrect, but probably not too incorrect
            // and I'm not sure what to actually do with Ty::Unknown
            // maybe an alternative would be `for<T> T`? (meaningless in rust, but expressible in chalk's Ty)
            Ty::Unknown => {
                PlaceholderIndex { ui: UniverseIndex::ROOT, idx: usize::max_value() }.to_ty()
            }
        }
    }
    fn from_chalk(db: &impl HirDatabase, chalk: chalk_ir::Ty) -> Self {
        match chalk {
            chalk_ir::Ty::Apply(apply_ty) => {
                match apply_ty.name {
                    TypeName::TypeKindId(TypeKindId::StructId(struct_id)) => {
                        let ctor = from_chalk(db, struct_id);
                        let parameters = from_chalk(db, apply_ty.parameters);
                        Ty::Apply(ApplicationTy { ctor, parameters })
                    }
                    // FIXME handle TypeKindId::Trait/Type here
                    TypeName::TypeKindId(_) => unimplemented!(),
                    TypeName::AssociatedType(_) => unimplemented!(),
                    TypeName::Placeholder(idx) => {
                        assert_eq!(idx.ui, UniverseIndex::ROOT);
                        Ty::Param { idx: idx.idx as u32, name: crate::Name::missing() }
                    }
                }
            }
            chalk_ir::Ty::Projection(_) => unimplemented!(),
            chalk_ir::Ty::UnselectedProjection(_) => unimplemented!(),
            chalk_ir::Ty::ForAll(_) => unimplemented!(),
            chalk_ir::Ty::BoundVar(idx) => Ty::Bound(idx as u32),
            chalk_ir::Ty::InferenceVar(_iv) => panic!("unexpected chalk infer ty"),
        }
    }
}

impl ToChalk for Substs {
    type Chalk = Vec<chalk_ir::Parameter>;

    fn to_chalk(self, db: &impl HirDatabase) -> Vec<Parameter> {
        self.iter().map(|ty| ty.clone().to_chalk(db).cast()).collect()
    }

    fn from_chalk(db: &impl HirDatabase, parameters: Vec<chalk_ir::Parameter>) -> Substs {
        parameters
            .into_iter()
            .map(|p| match p {
                chalk_ir::Parameter(chalk_ir::ParameterKind::Ty(ty)) => from_chalk(db, ty),
                chalk_ir::Parameter(chalk_ir::ParameterKind::Lifetime(_)) => unimplemented!(),
            })
            .collect::<Vec<_>>()
            .into()
    }
}

impl ToChalk for TraitRef {
    type Chalk = chalk_ir::TraitRef;

    fn to_chalk(self: TraitRef, db: &impl HirDatabase) -> chalk_ir::TraitRef {
        let trait_id = self.trait_.to_chalk(db);
        let parameters = self.substs.to_chalk(db);
        chalk_ir::TraitRef { trait_id, parameters }
    }

    fn from_chalk(db: &impl HirDatabase, trait_ref: chalk_ir::TraitRef) -> Self {
        let trait_ = from_chalk(db, trait_ref.trait_id);
        let substs = from_chalk(db, trait_ref.parameters);
        TraitRef { trait_, substs }
    }
}

impl ToChalk for Trait {
    type Chalk = chalk_ir::TraitId;

    fn to_chalk(self, _db: &impl HirDatabase) -> chalk_ir::TraitId {
        self.id.into()
    }

    fn from_chalk(_db: &impl HirDatabase, trait_id: chalk_ir::TraitId) -> Trait {
        Trait { id: trait_id.into() }
    }
}

impl ToChalk for TypeCtor {
    type Chalk = chalk_ir::StructId;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::StructId {
        db.intern_type_ctor(self).into()
    }

    fn from_chalk(db: &impl HirDatabase, struct_id: chalk_ir::StructId) -> TypeCtor {
        db.lookup_intern_type_ctor(struct_id.into())
    }
}

impl ToChalk for ImplBlock {
    type Chalk = chalk_ir::ImplId;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::ImplId {
        db.intern_impl_block(self).into()
    }

    fn from_chalk(db: &impl HirDatabase, impl_id: chalk_ir::ImplId) -> ImplBlock {
        db.lookup_intern_impl_block(impl_id.into())
    }
}

impl ToChalk for GenericPredicate {
    type Chalk = chalk_ir::QuantifiedWhereClause;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::QuantifiedWhereClause {
        match self {
            GenericPredicate::Implemented(trait_ref) => {
                make_binders(chalk_ir::WhereClause::Implemented(trait_ref.to_chalk(db)), 0)
            }
            GenericPredicate::Error => {
                let impossible_trait_ref = chalk_ir::TraitRef {
                    trait_id: UNKNOWN_TRAIT,
                    parameters: vec![Ty::Unknown.to_chalk(db).cast()],
                };
                make_binders(chalk_ir::WhereClause::Implemented(impossible_trait_ref), 0)
            }
        }
    }

    fn from_chalk(
        _db: &impl HirDatabase,
        _where_clause: chalk_ir::QuantifiedWhereClause,
    ) -> GenericPredicate {
        // This should never need to be called
        unimplemented!()
    }
}

fn make_binders<T>(value: T, num_vars: usize) -> chalk_ir::Binders<T> {
    chalk_ir::Binders {
        value,
        binders: std::iter::repeat(chalk_ir::ParameterKind::Ty(())).take(num_vars).collect(),
    }
}

fn blacklisted_trait(db: &impl HirDatabase, trait_: Trait) -> bool {
    let name = trait_.name(db).unwrap_or_else(crate::Name::missing).to_string();
    match &*name {
        "Send" | "Sync" | "Sized" | "Fn" | "FnMut" | "FnOnce" => true,
        _ => false,
    }
}

fn convert_where_clauses(
    db: &impl HirDatabase,
    def: GenericDef,
    substs: &Substs,
) -> Vec<chalk_ir::QuantifiedWhereClause> {
    let generic_predicates = db.generic_predicates(def);
    let mut result = Vec::with_capacity(generic_predicates.len());
    for pred in generic_predicates.iter() {
        if pred.is_error() {
            // HACK: Return just the single predicate (which is always false
            // anyway), otherwise Chalk can easily get into slow situations
            return vec![pred.clone().subst(substs).to_chalk(db)];
        }
        if let GenericPredicate::Implemented(trait_ref) = pred {
            if blacklisted_trait(db, trait_ref.trait_) {
                continue;
            }
        }
        result.push(pred.clone().subst(substs).to_chalk(db));
    }
    result
}

impl<'a, DB> chalk_solve::RustIrDatabase for ChalkContext<'a, DB>
where
    DB: HirDatabase,
{
    fn associated_ty_data(&self, _ty: TypeId) -> Arc<AssociatedTyDatum> {
        unimplemented!()
    }
    fn trait_datum(&self, trait_id: chalk_ir::TraitId) -> Arc<TraitDatum> {
        debug!("trait_datum {:?}", trait_id);
        if trait_id == UNKNOWN_TRAIT {
            let trait_datum_bound = chalk_rust_ir::TraitDatumBound {
                trait_ref: chalk_ir::TraitRef {
                    trait_id: UNKNOWN_TRAIT,
                    parameters: vec![chalk_ir::Ty::BoundVar(0).cast()],
                },
                associated_ty_ids: Vec::new(),
                where_clauses: Vec::new(),
                flags: chalk_rust_ir::TraitFlags {
                    auto: false,
                    marker: false,
                    upstream: true,
                    fundamental: false,
                },
            };
            return Arc::new(TraitDatum { binders: make_binders(trait_datum_bound, 1) });
        }
        let trait_: Trait = from_chalk(self.db, trait_id);
        debug!("trait {:?} = {:?}", trait_id, trait_.name(self.db));
        let generic_params = trait_.generic_params(self.db);
        let bound_vars = Substs::bound_vars(&generic_params);
        let trait_ref = trait_.trait_ref(self.db).subst(&bound_vars).to_chalk(self.db);
        let flags = chalk_rust_ir::TraitFlags {
            auto: trait_.is_auto(self.db),
            upstream: trait_.module(self.db).krate(self.db) != Some(self.krate),
            // FIXME set these flags correctly
            marker: false,
            fundamental: false,
        };
        let where_clauses = convert_where_clauses(self.db, trait_.into(), &bound_vars);
        let associated_ty_ids = Vec::new(); // FIXME add associated tys
        let trait_datum_bound =
            chalk_rust_ir::TraitDatumBound { trait_ref, where_clauses, flags, associated_ty_ids };
        let trait_datum = TraitDatum { binders: make_binders(trait_datum_bound, bound_vars.len()) };
        Arc::new(trait_datum)
    }
    fn struct_datum(&self, struct_id: chalk_ir::StructId) -> Arc<StructDatum> {
        debug!("struct_datum {:?}", struct_id);
        let type_ctor = from_chalk(self.db, struct_id);
        debug!("struct {:?} = {:?}", struct_id, type_ctor);
        // FIXME might be nicer if we can create a fake GenericParams for the TypeCtor
        // FIXME extract this to a method on Ty
        let (num_params, where_clauses, upstream) = match type_ctor {
            TypeCtor::Bool
            | TypeCtor::Char
            | TypeCtor::Int(_)
            | TypeCtor::Float(_)
            | TypeCtor::Never
            | TypeCtor::Str => (0, vec![], true),
            TypeCtor::Slice | TypeCtor::Array | TypeCtor::RawPtr(_) | TypeCtor::Ref(_) => {
                (1, vec![], true)
            }
            TypeCtor::FnPtr { num_args } => (num_args as usize + 1, vec![], true),
            TypeCtor::Tuple { cardinality } => (cardinality as usize, vec![], true),
            TypeCtor::FnDef(callable) => {
                tested_by!(trait_resolution_on_fn_type);
                let krate = match callable {
                    CallableDef::Function(f) => f.module(self.db).krate(self.db),
                    CallableDef::Struct(s) => s.module(self.db).krate(self.db),
                    CallableDef::EnumVariant(v) => {
                        v.parent_enum(self.db).module(self.db).krate(self.db)
                    }
                };
                let generic_def: GenericDef = match callable {
                    CallableDef::Function(f) => f.into(),
                    CallableDef::Struct(s) => s.into(),
                    CallableDef::EnumVariant(v) => v.parent_enum(self.db).into(),
                };
                let generic_params = generic_def.generic_params(self.db);
                let bound_vars = Substs::bound_vars(&generic_params);
                let where_clauses = convert_where_clauses(self.db, generic_def, &bound_vars);
                (
                    generic_params.count_params_including_parent(),
                    where_clauses,
                    krate != Some(self.krate),
                )
            }
            TypeCtor::Adt(adt) => {
                let generic_params = adt.generic_params(self.db);
                let bound_vars = Substs::bound_vars(&generic_params);
                let where_clauses = convert_where_clauses(self.db, adt.into(), &bound_vars);
                (
                    generic_params.count_params_including_parent(),
                    where_clauses,
                    adt.krate(self.db) != Some(self.krate),
                )
            }
        };
        let flags = chalk_rust_ir::StructFlags {
            upstream,
            // FIXME set fundamental flag correctly
            fundamental: false,
        };
        let self_ty = chalk_ir::ApplicationTy {
            name: TypeName::TypeKindId(type_ctor.to_chalk(self.db).into()),
            parameters: (0..num_params).map(|i| chalk_ir::Ty::BoundVar(i).cast()).collect(),
        };
        let struct_datum_bound = chalk_rust_ir::StructDatumBound {
            self_ty,
            fields: Vec::new(), // FIXME add fields (only relevant for auto traits)
            where_clauses,
            flags,
        };
        let struct_datum = StructDatum { binders: make_binders(struct_datum_bound, num_params) };
        Arc::new(struct_datum)
    }
    fn impl_datum(&self, impl_id: ImplId) -> Arc<ImplDatum> {
        debug!("impl_datum {:?}", impl_id);
        let impl_block: ImplBlock = from_chalk(self.db, impl_id);
        let generic_params = impl_block.generic_params(self.db);
        let bound_vars = Substs::bound_vars(&generic_params);
        let trait_ref = impl_block
            .target_trait_ref(self.db)
            .expect("FIXME handle unresolved impl block trait ref")
            .subst(&bound_vars);
        let impl_type = if impl_block.module().krate(self.db) == Some(self.krate) {
            chalk_rust_ir::ImplType::Local
        } else {
            chalk_rust_ir::ImplType::External
        };
        let where_clauses = convert_where_clauses(self.db, impl_block.into(), &bound_vars);
        let negative = impl_block.is_negative(self.db);
        debug!(
            "impl {:?}: {}{} where {:?}",
            impl_id,
            if negative { "!" } else { "" },
            trait_ref.display(self.db),
            where_clauses
        );
        let trait_ref = trait_ref.to_chalk(self.db);
        let impl_datum_bound = chalk_rust_ir::ImplDatumBound {
            trait_ref: if negative {
                chalk_rust_ir::PolarizedTraitRef::Negative(trait_ref)
            } else {
                chalk_rust_ir::PolarizedTraitRef::Positive(trait_ref)
            },
            where_clauses,
            associated_ty_values: Vec::new(), // FIXME add associated type values
            impl_type,
        };
        let impl_datum = ImplDatum { binders: make_binders(impl_datum_bound, bound_vars.len()) };
        Arc::new(impl_datum)
    }
    fn impls_for_trait(&self, trait_id: chalk_ir::TraitId) -> Vec<ImplId> {
        debug!("impls_for_trait {:?}", trait_id);
        if trait_id == UNKNOWN_TRAIT {
            return Vec::new();
        }
        let trait_: Trait = from_chalk(self.db, trait_id);
        let blacklisted = blacklisted_trait(self.db, trait_);
        if blacklisted {
            return Vec::new();
        }
        let result: Vec<_> = self
            .db
            .impls_for_trait(self.krate, trait_)
            .iter()
            .map(|impl_block| impl_block.to_chalk(self.db))
            .collect();
        debug!("impls_for_trait returned {} impls", result.len());
        result
    }
    fn impl_provided_for(
        &self,
        auto_trait_id: chalk_ir::TraitId,
        struct_id: chalk_ir::StructId,
    ) -> bool {
        debug!("impl_provided_for {:?}, {:?}", auto_trait_id, struct_id);
        false // FIXME
    }
    fn type_name(&self, _id: TypeKindId) -> Identifier {
        unimplemented!()
    }
    fn split_projection<'p>(
        &self,
        projection: &'p ProjectionTy,
    ) -> (Arc<AssociatedTyDatum>, &'p [Parameter], &'p [Parameter]) {
        debug!("split_projection {:?}", projection);
        unimplemented!()
    }
    fn custom_clauses(&self) -> Vec<chalk_ir::ProgramClause> {
        debug!("custom_clauses");
        vec![]
    }
    fn all_structs(&self) -> Vec<chalk_ir::StructId> {
        debug!("all_structs");
        // FIXME
        vec![]
    }
}

fn id_from_chalk<T: InternKey>(chalk_id: chalk_ir::RawId) -> T {
    T::from_intern_id(InternId::from(chalk_id.index))
}
fn id_to_chalk<T: InternKey>(salsa_id: T) -> chalk_ir::RawId {
    chalk_ir::RawId { index: salsa_id.as_intern_id().as_u32() }
}

impl From<chalk_ir::TraitId> for crate::ids::TraitId {
    fn from(trait_id: chalk_ir::TraitId) -> Self {
        id_from_chalk(trait_id.0)
    }
}

impl From<crate::ids::TraitId> for chalk_ir::TraitId {
    fn from(trait_id: crate::ids::TraitId) -> Self {
        chalk_ir::TraitId(id_to_chalk(trait_id))
    }
}

impl From<chalk_ir::StructId> for crate::ids::TypeCtorId {
    fn from(struct_id: chalk_ir::StructId) -> Self {
        id_from_chalk(struct_id.0)
    }
}

impl From<crate::ids::TypeCtorId> for chalk_ir::StructId {
    fn from(type_ctor_id: crate::ids::TypeCtorId) -> Self {
        chalk_ir::StructId(id_to_chalk(type_ctor_id))
    }
}

impl From<chalk_ir::ImplId> for crate::ids::GlobalImplId {
    fn from(impl_id: chalk_ir::ImplId) -> Self {
        id_from_chalk(impl_id.0)
    }
}

impl From<crate::ids::GlobalImplId> for chalk_ir::ImplId {
    fn from(impl_id: crate::ids::GlobalImplId) -> Self {
        chalk_ir::ImplId(id_to_chalk(impl_id))
    }
}
