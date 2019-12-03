//! Conversion code from/to Chalk.
use std::sync::Arc;

use log::debug;

use chalk_ir::{
    cast::Cast, family::ChalkIr, Identifier, Parameter, PlaceholderIndex, TypeId, TypeKindId,
    TypeName, UniverseIndex,
};
use chalk_rust_ir::{AssociatedTyDatum, AssociatedTyValue, ImplDatum, StructDatum, TraitDatum};

use hir_def::{
    AssocItemId, AstItemDef, ContainerId, GenericDefId, ImplId, Lookup, TraitId, TypeAliasId,
};
use ra_db::{
    salsa::{InternId, InternKey},
    CrateId,
};

use super::{builtin, AssocTyValue, Canonical, ChalkContext, Impl, Obligation};
use crate::{
    db::HirDatabase, display::HirDisplay, ApplicationTy, GenericPredicate, ProjectionTy, Substs,
    TraitRef, Ty, TypeCtor, TypeWalk,
};

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
    type Chalk = chalk_ir::Ty<ChalkIr>;
    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::Ty<ChalkIr> {
        match self {
            Ty::Apply(apply_ty) => {
                let name = match apply_ty.ctor {
                    TypeCtor::AssociatedType(type_alias) => {
                        let type_id = type_alias.to_chalk(db);
                        TypeName::AssociatedType(type_id)
                    }
                    _ => {
                        // other TypeCtors get interned and turned into a chalk StructId
                        let struct_id = apply_ty.ctor.to_chalk(db);
                        TypeName::TypeKindId(struct_id.into())
                    }
                };
                let parameters = apply_ty.parameters.to_chalk(db);
                chalk_ir::ApplicationTy { name, parameters }.cast().intern()
            }
            Ty::Projection(proj_ty) => {
                let associated_ty_id = proj_ty.associated_ty.to_chalk(db);
                let parameters = proj_ty.parameters.to_chalk(db);
                chalk_ir::ProjectionTy { associated_ty_id, parameters }.cast().intern()
            }
            Ty::Param { idx, .. } => {
                PlaceholderIndex { ui: UniverseIndex::ROOT, idx: idx as usize }.to_ty::<ChalkIr>()
            }
            Ty::Bound(idx) => chalk_ir::TyData::BoundVar(idx as usize).intern(),
            Ty::Infer(_infer_ty) => panic!("uncanonicalized infer ty"),
            Ty::Dyn(predicates) => {
                let where_clauses = predicates.iter().cloned().map(|p| p.to_chalk(db)).collect();
                chalk_ir::TyData::Dyn(make_binders(where_clauses, 1)).intern()
            }
            Ty::Opaque(predicates) => {
                let where_clauses = predicates.iter().cloned().map(|p| p.to_chalk(db)).collect();
                chalk_ir::TyData::Opaque(make_binders(where_clauses, 1)).intern()
            }
            Ty::Unknown => {
                let parameters = Vec::new();
                let name = TypeName::Error;
                chalk_ir::ApplicationTy { name, parameters }.cast().intern()
            }
        }
    }
    fn from_chalk(db: &impl HirDatabase, chalk: chalk_ir::Ty<ChalkIr>) -> Self {
        match chalk.data().clone() {
            chalk_ir::TyData::Apply(apply_ty) => {
                // FIXME this is kind of hacky due to the fact that
                // TypeName::Placeholder is a Ty::Param on our side
                match apply_ty.name {
                    TypeName::TypeKindId(TypeKindId::StructId(struct_id)) => {
                        let ctor = from_chalk(db, struct_id);
                        let parameters = from_chalk(db, apply_ty.parameters);
                        Ty::Apply(ApplicationTy { ctor, parameters })
                    }
                    TypeName::AssociatedType(type_id) => {
                        let ctor = TypeCtor::AssociatedType(from_chalk(db, type_id));
                        let parameters = from_chalk(db, apply_ty.parameters);
                        Ty::Apply(ApplicationTy { ctor, parameters })
                    }
                    TypeName::Error => Ty::Unknown,
                    // FIXME handle TypeKindId::Trait/Type here
                    TypeName::TypeKindId(_) => unimplemented!(),
                    TypeName::Placeholder(idx) => {
                        assert_eq!(idx.ui, UniverseIndex::ROOT);
                        Ty::Param { idx: idx.idx as u32, name: crate::Name::missing() }
                    }
                }
            }
            chalk_ir::TyData::Projection(proj) => {
                let associated_ty = from_chalk(db, proj.associated_ty_id);
                let parameters = from_chalk(db, proj.parameters);
                Ty::Projection(ProjectionTy { associated_ty, parameters })
            }
            chalk_ir::TyData::ForAll(_) => unimplemented!(),
            chalk_ir::TyData::BoundVar(idx) => Ty::Bound(idx as u32),
            chalk_ir::TyData::InferenceVar(_iv) => Ty::Unknown,
            chalk_ir::TyData::Dyn(where_clauses) => {
                assert_eq!(where_clauses.binders.len(), 1);
                let predicates =
                    where_clauses.value.into_iter().map(|c| from_chalk(db, c)).collect();
                Ty::Dyn(predicates)
            }
            chalk_ir::TyData::Opaque(where_clauses) => {
                assert_eq!(where_clauses.binders.len(), 1);
                let predicates =
                    where_clauses.value.into_iter().map(|c| from_chalk(db, c)).collect();
                Ty::Opaque(predicates)
            }
        }
    }
}

impl ToChalk for Substs {
    type Chalk = Vec<chalk_ir::Parameter<ChalkIr>>;

    fn to_chalk(self, db: &impl HirDatabase) -> Vec<Parameter<ChalkIr>> {
        self.iter().map(|ty| ty.clone().to_chalk(db).cast()).collect()
    }

    fn from_chalk(db: &impl HirDatabase, parameters: Vec<chalk_ir::Parameter<ChalkIr>>) -> Substs {
        let tys = parameters
            .into_iter()
            .map(|p| match p {
                chalk_ir::Parameter(chalk_ir::ParameterKind::Ty(ty)) => from_chalk(db, ty),
                chalk_ir::Parameter(chalk_ir::ParameterKind::Lifetime(_)) => unimplemented!(),
            })
            .collect();
        Substs(tys)
    }
}

impl ToChalk for TraitRef {
    type Chalk = chalk_ir::TraitRef<ChalkIr>;

    fn to_chalk(self: TraitRef, db: &impl HirDatabase) -> chalk_ir::TraitRef<ChalkIr> {
        let trait_id = self.trait_.to_chalk(db);
        let parameters = self.substs.to_chalk(db);
        chalk_ir::TraitRef { trait_id, parameters }
    }

    fn from_chalk(db: &impl HirDatabase, trait_ref: chalk_ir::TraitRef<ChalkIr>) -> Self {
        let trait_ = from_chalk(db, trait_ref.trait_id);
        let substs = from_chalk(db, trait_ref.parameters);
        TraitRef { trait_, substs }
    }
}

impl ToChalk for TraitId {
    type Chalk = chalk_ir::TraitId;

    fn to_chalk(self, _db: &impl HirDatabase) -> chalk_ir::TraitId {
        chalk_ir::TraitId(id_to_chalk(self))
    }

    fn from_chalk(_db: &impl HirDatabase, trait_id: chalk_ir::TraitId) -> TraitId {
        id_from_chalk(trait_id.0)
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

impl ToChalk for Impl {
    type Chalk = chalk_ir::ImplId;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::ImplId {
        db.intern_chalk_impl(self).into()
    }

    fn from_chalk(db: &impl HirDatabase, impl_id: chalk_ir::ImplId) -> Impl {
        db.lookup_intern_chalk_impl(impl_id.into())
    }
}

impl ToChalk for TypeAliasId {
    type Chalk = chalk_ir::TypeId;

    fn to_chalk(self, _db: &impl HirDatabase) -> chalk_ir::TypeId {
        chalk_ir::TypeId(id_to_chalk(self))
    }

    fn from_chalk(_db: &impl HirDatabase, type_alias_id: chalk_ir::TypeId) -> TypeAliasId {
        id_from_chalk(type_alias_id.0)
    }
}

impl ToChalk for AssocTyValue {
    type Chalk = chalk_rust_ir::AssociatedTyValueId;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_rust_ir::AssociatedTyValueId {
        db.intern_assoc_ty_value(self).into()
    }

    fn from_chalk(
        db: &impl HirDatabase,
        assoc_ty_value_id: chalk_rust_ir::AssociatedTyValueId,
    ) -> AssocTyValue {
        db.lookup_intern_assoc_ty_value(assoc_ty_value_id.into())
    }
}

impl ToChalk for GenericPredicate {
    type Chalk = chalk_ir::QuantifiedWhereClause<ChalkIr>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::QuantifiedWhereClause<ChalkIr> {
        match self {
            GenericPredicate::Implemented(trait_ref) => {
                make_binders(chalk_ir::WhereClause::Implemented(trait_ref.to_chalk(db)), 0)
            }
            GenericPredicate::Projection(projection_pred) => make_binders(
                chalk_ir::WhereClause::ProjectionEq(chalk_ir::ProjectionEq {
                    projection: projection_pred.projection_ty.to_chalk(db),
                    ty: projection_pred.ty.to_chalk(db),
                }),
                0,
            ),
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
        db: &impl HirDatabase,
        where_clause: chalk_ir::QuantifiedWhereClause<ChalkIr>,
    ) -> GenericPredicate {
        match where_clause.value {
            chalk_ir::WhereClause::Implemented(tr) => {
                if tr.trait_id == UNKNOWN_TRAIT {
                    // FIXME we need an Error enum on the Chalk side to avoid this
                    return GenericPredicate::Error;
                }
                GenericPredicate::Implemented(from_chalk(db, tr))
            }
            chalk_ir::WhereClause::ProjectionEq(projection_eq) => {
                let projection_ty = from_chalk(db, projection_eq.projection);
                let ty = from_chalk(db, projection_eq.ty);
                GenericPredicate::Projection(super::ProjectionPredicate { projection_ty, ty })
            }
        }
    }
}

impl ToChalk for ProjectionTy {
    type Chalk = chalk_ir::ProjectionTy<ChalkIr>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::ProjectionTy<ChalkIr> {
        chalk_ir::ProjectionTy {
            associated_ty_id: self.associated_ty.to_chalk(db),
            parameters: self.parameters.to_chalk(db),
        }
    }

    fn from_chalk(
        db: &impl HirDatabase,
        projection_ty: chalk_ir::ProjectionTy<ChalkIr>,
    ) -> ProjectionTy {
        ProjectionTy {
            associated_ty: from_chalk(db, projection_ty.associated_ty_id),
            parameters: from_chalk(db, projection_ty.parameters),
        }
    }
}

impl ToChalk for super::ProjectionPredicate {
    type Chalk = chalk_ir::Normalize<ChalkIr>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::Normalize<ChalkIr> {
        chalk_ir::Normalize {
            projection: self.projection_ty.to_chalk(db),
            ty: self.ty.to_chalk(db),
        }
    }

    fn from_chalk(_db: &impl HirDatabase, _normalize: chalk_ir::Normalize<ChalkIr>) -> Self {
        unimplemented!()
    }
}

impl ToChalk for Obligation {
    type Chalk = chalk_ir::DomainGoal<ChalkIr>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::DomainGoal<ChalkIr> {
        match self {
            Obligation::Trait(tr) => tr.to_chalk(db).cast(),
            Obligation::Projection(pr) => pr.to_chalk(db).cast(),
        }
    }

    fn from_chalk(_db: &impl HirDatabase, _goal: chalk_ir::DomainGoal<ChalkIr>) -> Self {
        unimplemented!()
    }
}

impl<T> ToChalk for Canonical<T>
where
    T: ToChalk,
{
    type Chalk = chalk_ir::Canonical<T::Chalk>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::Canonical<T::Chalk> {
        let parameter = chalk_ir::ParameterKind::Ty(chalk_ir::UniverseIndex::ROOT);
        let value = self.value.to_chalk(db);
        let canonical = chalk_ir::Canonical { value, binders: vec![parameter; self.num_vars] };
        canonical
    }

    fn from_chalk(db: &impl HirDatabase, canonical: chalk_ir::Canonical<T::Chalk>) -> Canonical<T> {
        Canonical { num_vars: canonical.binders.len(), value: from_chalk(db, canonical.value) }
    }
}

impl ToChalk for Arc<super::TraitEnvironment> {
    type Chalk = chalk_ir::Environment<ChalkIr>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::Environment<ChalkIr> {
        let mut clauses = Vec::new();
        for pred in &self.predicates {
            if pred.is_error() {
                // for env, we just ignore errors
                continue;
            }
            let program_clause: chalk_ir::ProgramClause<ChalkIr> = pred.clone().to_chalk(db).cast();
            clauses.push(program_clause.into_from_env_clause());
        }
        chalk_ir::Environment::new().add_clauses(clauses)
    }

    fn from_chalk(
        _db: &impl HirDatabase,
        _env: chalk_ir::Environment<ChalkIr>,
    ) -> Arc<super::TraitEnvironment> {
        unimplemented!()
    }
}

impl<T: ToChalk> ToChalk for super::InEnvironment<T>
where
    T::Chalk: chalk_ir::family::HasTypeFamily<TypeFamily = ChalkIr>,
{
    type Chalk = chalk_ir::InEnvironment<T::Chalk>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::InEnvironment<T::Chalk> {
        chalk_ir::InEnvironment {
            environment: self.environment.to_chalk(db),
            goal: self.value.to_chalk(db),
        }
    }

    fn from_chalk(
        db: &impl HirDatabase,
        in_env: chalk_ir::InEnvironment<T::Chalk>,
    ) -> super::InEnvironment<T> {
        super::InEnvironment {
            environment: from_chalk(db, in_env.environment),
            value: from_chalk(db, in_env.goal),
        }
    }
}

impl ToChalk for builtin::BuiltinImplData {
    type Chalk = chalk_rust_ir::ImplDatum<ChalkIr>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_rust_ir::ImplDatum<ChalkIr> {
        let impl_type = chalk_rust_ir::ImplType::External;
        let where_clauses = self.where_clauses.into_iter().map(|w| w.to_chalk(db)).collect();

        let impl_datum_bound =
            chalk_rust_ir::ImplDatumBound { trait_ref: self.trait_ref.to_chalk(db), where_clauses };
        let associated_ty_value_ids =
            self.assoc_ty_values.into_iter().map(|v| v.to_chalk(db)).collect();
        chalk_rust_ir::ImplDatum {
            binders: make_binders(impl_datum_bound, self.num_vars),
            impl_type,
            polarity: chalk_rust_ir::Polarity::Positive,
            associated_ty_value_ids,
        }
    }

    fn from_chalk(_db: &impl HirDatabase, _data: chalk_rust_ir::ImplDatum<ChalkIr>) -> Self {
        unimplemented!()
    }
}

impl ToChalk for builtin::BuiltinImplAssocTyValueData {
    type Chalk = chalk_rust_ir::AssociatedTyValue<ChalkIr>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_rust_ir::AssociatedTyValue<ChalkIr> {
        let value_bound = chalk_rust_ir::AssociatedTyValueBound { ty: self.value.to_chalk(db) };

        chalk_rust_ir::AssociatedTyValue {
            associated_ty_id: self.assoc_ty_id.to_chalk(db),
            impl_id: self.impl_.to_chalk(db),
            value: make_binders(value_bound, self.num_vars),
        }
    }

    fn from_chalk(
        _db: &impl HirDatabase,
        _data: chalk_rust_ir::AssociatedTyValue<ChalkIr>,
    ) -> builtin::BuiltinImplAssocTyValueData {
        unimplemented!()
    }
}

fn make_binders<T>(value: T, num_vars: usize) -> chalk_ir::Binders<T> {
    chalk_ir::Binders {
        value,
        binders: std::iter::repeat(chalk_ir::ParameterKind::Ty(())).take(num_vars).collect(),
    }
}

fn convert_where_clauses(
    db: &impl HirDatabase,
    def: GenericDefId,
    substs: &Substs,
) -> Vec<chalk_ir::QuantifiedWhereClause<ChalkIr>> {
    let generic_predicates = db.generic_predicates(def);
    let mut result = Vec::with_capacity(generic_predicates.len());
    for pred in generic_predicates.iter() {
        if pred.is_error() {
            // HACK: Return just the single predicate (which is always false
            // anyway), otherwise Chalk can easily get into slow situations
            return vec![pred.clone().subst(substs).to_chalk(db)];
        }
        result.push(pred.clone().subst(substs).to_chalk(db));
    }
    result
}

impl<'a, DB> chalk_solve::RustIrDatabase<ChalkIr> for ChalkContext<'a, DB>
where
    DB: HirDatabase,
{
    fn associated_ty_data(&self, id: TypeId) -> Arc<AssociatedTyDatum<ChalkIr>> {
        self.db.associated_ty_data(id)
    }
    fn trait_datum(&self, trait_id: chalk_ir::TraitId) -> Arc<TraitDatum<ChalkIr>> {
        self.db.trait_datum(self.krate, trait_id)
    }
    fn struct_datum(&self, struct_id: chalk_ir::StructId) -> Arc<StructDatum<ChalkIr>> {
        self.db.struct_datum(self.krate, struct_id)
    }
    fn impl_datum(&self, impl_id: chalk_ir::ImplId) -> Arc<ImplDatum<ChalkIr>> {
        self.db.impl_datum(self.krate, impl_id)
    }
    fn impls_for_trait(
        &self,
        trait_id: chalk_ir::TraitId,
        parameters: &[Parameter<ChalkIr>],
    ) -> Vec<chalk_ir::ImplId> {
        debug!("impls_for_trait {:?}", trait_id);
        if trait_id == UNKNOWN_TRAIT {
            return Vec::new();
        }
        let trait_: TraitId = from_chalk(self.db, trait_id);
        let mut result: Vec<_> = self
            .db
            .impls_for_trait(self.krate, trait_.into())
            .iter()
            .copied()
            .map(|it| Impl::ImplBlock(it.into()))
            .map(|impl_| impl_.to_chalk(self.db))
            .collect();

        let ty: Ty = from_chalk(self.db, parameters[0].assert_ty_ref().clone());

        builtin::get_builtin_impls(self.db, self.krate, &ty, trait_, |i| {
            result.push(i.to_chalk(self.db))
        });

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
    fn associated_ty_value(
        &self,
        id: chalk_rust_ir::AssociatedTyValueId,
    ) -> Arc<AssociatedTyValue<ChalkIr>> {
        self.db.associated_ty_value(self.krate.into(), id)
    }
    fn custom_clauses(&self) -> Vec<chalk_ir::ProgramClause<ChalkIr>> {
        vec![]
    }
    fn local_impls_to_coherence_check(
        &self,
        _trait_id: chalk_ir::TraitId,
    ) -> Vec<chalk_ir::ImplId> {
        // We don't do coherence checking (yet)
        unimplemented!()
    }
}

pub(crate) fn associated_ty_data_query(
    db: &impl HirDatabase,
    id: TypeId,
) -> Arc<AssociatedTyDatum<ChalkIr>> {
    debug!("associated_ty_data {:?}", id);
    let type_alias: TypeAliasId = from_chalk(db, id);
    let trait_ = match type_alias.lookup(db).container {
        ContainerId::TraitId(t) => t,
        _ => panic!("associated type not in trait"),
    };
    let generic_params = db.generic_params(type_alias.into());
    let bound_data = chalk_rust_ir::AssociatedTyDatumBound {
        // FIXME add bounds and where clauses
        bounds: vec![],
        where_clauses: vec![],
    };
    let datum = AssociatedTyDatum {
        trait_id: trait_.to_chalk(db),
        id,
        name: lalrpop_intern::intern(&db.type_alias_data(type_alias).name.to_string()),
        binders: make_binders(bound_data, generic_params.count_params_including_parent()),
    };
    Arc::new(datum)
}

pub(crate) fn trait_datum_query(
    db: &impl HirDatabase,
    krate: CrateId,
    trait_id: chalk_ir::TraitId,
) -> Arc<TraitDatum<ChalkIr>> {
    debug!("trait_datum {:?}", trait_id);
    if trait_id == UNKNOWN_TRAIT {
        let trait_datum_bound = chalk_rust_ir::TraitDatumBound { where_clauses: Vec::new() };

        let flags = chalk_rust_ir::TraitFlags {
            auto: false,
            marker: false,
            upstream: true,
            fundamental: false,
            non_enumerable: true,
            coinductive: false,
        };
        return Arc::new(TraitDatum {
            id: trait_id,
            binders: make_binders(trait_datum_bound, 1),
            flags,
            associated_ty_ids: vec![],
        });
    }
    let trait_: TraitId = from_chalk(db, trait_id);
    let trait_data = db.trait_data(trait_);
    debug!("trait {:?} = {:?}", trait_id, trait_data.name);
    let generic_params = db.generic_params(trait_.into());
    let bound_vars = Substs::bound_vars(&generic_params);
    let flags = chalk_rust_ir::TraitFlags {
        auto: trait_data.auto,
        upstream: trait_.module(db).krate != krate,
        non_enumerable: true,
        coinductive: false, // only relevant for Chalk testing
        // FIXME set these flags correctly
        marker: false,
        fundamental: false,
    };
    let where_clauses = convert_where_clauses(db, trait_.into(), &bound_vars);
    let associated_ty_ids =
        trait_data.associated_types().map(|type_alias| type_alias.to_chalk(db)).collect();
    let trait_datum_bound = chalk_rust_ir::TraitDatumBound { where_clauses };
    let trait_datum = TraitDatum {
        id: trait_id,
        binders: make_binders(trait_datum_bound, bound_vars.len()),
        flags,
        associated_ty_ids,
    };
    Arc::new(trait_datum)
}

pub(crate) fn struct_datum_query(
    db: &impl HirDatabase,
    krate: CrateId,
    struct_id: chalk_ir::StructId,
) -> Arc<StructDatum<ChalkIr>> {
    debug!("struct_datum {:?}", struct_id);
    let type_ctor: TypeCtor = from_chalk(db, struct_id);
    debug!("struct {:?} = {:?}", struct_id, type_ctor);
    let num_params = type_ctor.num_ty_params(db);
    let upstream = type_ctor.krate(db) != Some(krate);
    let where_clauses = type_ctor
        .as_generic_def()
        .map(|generic_def| {
            let generic_params = db.generic_params(generic_def.into());
            let bound_vars = Substs::bound_vars(&generic_params);
            convert_where_clauses(db, generic_def, &bound_vars)
        })
        .unwrap_or_else(Vec::new);
    let flags = chalk_rust_ir::StructFlags {
        upstream,
        // FIXME set fundamental flag correctly
        fundamental: false,
    };
    let struct_datum_bound = chalk_rust_ir::StructDatumBound {
        fields: Vec::new(), // FIXME add fields (only relevant for auto traits)
        where_clauses,
    };
    let struct_datum =
        StructDatum { id: struct_id, binders: make_binders(struct_datum_bound, num_params), flags };
    Arc::new(struct_datum)
}

pub(crate) fn impl_datum_query(
    db: &impl HirDatabase,
    krate: CrateId,
    impl_id: chalk_ir::ImplId,
) -> Arc<ImplDatum<ChalkIr>> {
    let _p = ra_prof::profile("impl_datum");
    debug!("impl_datum {:?}", impl_id);
    let impl_: Impl = from_chalk(db, impl_id);
    match impl_ {
        Impl::ImplBlock(impl_block) => impl_block_datum(db, krate, impl_id, impl_block),
        _ => builtin::impl_datum(db, krate, impl_).map(|d| Arc::new(d.to_chalk(db))),
    }
    .unwrap_or_else(invalid_impl_datum)
}

fn impl_block_datum(
    db: &impl HirDatabase,
    krate: CrateId,
    chalk_id: chalk_ir::ImplId,
    impl_id: ImplId,
) -> Option<Arc<ImplDatum<ChalkIr>>> {
    let trait_ref = db.impl_trait(impl_id)?;
    let impl_data = db.impl_data(impl_id);

    let generic_params = db.generic_params(impl_id.into());
    let bound_vars = Substs::bound_vars(&generic_params);
    let trait_ref = trait_ref.subst(&bound_vars);
    let trait_ = trait_ref.trait_;
    let impl_type = if impl_id.module(db).krate == krate {
        chalk_rust_ir::ImplType::Local
    } else {
        chalk_rust_ir::ImplType::External
    };
    let where_clauses = convert_where_clauses(db, impl_id.into(), &bound_vars);
    let negative = impl_data.is_negative;
    debug!(
        "impl {:?}: {}{} where {:?}",
        chalk_id,
        if negative { "!" } else { "" },
        trait_ref.display(db),
        where_clauses
    );
    let trait_ref = trait_ref.to_chalk(db);

    let polarity = if negative {
        chalk_rust_ir::Polarity::Negative
    } else {
        chalk_rust_ir::Polarity::Positive
    };

    let impl_datum_bound = chalk_rust_ir::ImplDatumBound { trait_ref, where_clauses };
    let trait_data = db.trait_data(trait_);
    let associated_ty_value_ids = impl_data
        .items
        .iter()
        .filter_map(|item| match item {
            AssocItemId::TypeAliasId(type_alias) => Some(*type_alias),
            _ => None,
        })
        .filter(|&type_alias| {
            // don't include associated types that don't exist in the trait
            let name = &db.type_alias_data(type_alias).name;
            trait_data.associated_type_by_name(name).is_some()
        })
        .map(|type_alias| AssocTyValue::TypeAlias(type_alias).to_chalk(db))
        .collect();
    debug!("impl_datum: {:?}", impl_datum_bound);
    let impl_datum = ImplDatum {
        binders: make_binders(impl_datum_bound, bound_vars.len()),
        impl_type,
        polarity,
        associated_ty_value_ids,
    };
    Some(Arc::new(impl_datum))
}

fn invalid_impl_datum() -> Arc<ImplDatum<ChalkIr>> {
    let trait_ref = chalk_ir::TraitRef {
        trait_id: UNKNOWN_TRAIT,
        parameters: vec![chalk_ir::TyData::BoundVar(0).cast().intern().cast()],
    };
    let impl_datum_bound = chalk_rust_ir::ImplDatumBound { trait_ref, where_clauses: Vec::new() };
    let impl_datum = ImplDatum {
        binders: make_binders(impl_datum_bound, 1),
        impl_type: chalk_rust_ir::ImplType::External,
        polarity: chalk_rust_ir::Polarity::Positive,
        associated_ty_value_ids: Vec::new(),
    };
    Arc::new(impl_datum)
}

pub(crate) fn associated_ty_value_query(
    db: &impl HirDatabase,
    krate: CrateId,
    id: chalk_rust_ir::AssociatedTyValueId,
) -> Arc<chalk_rust_ir::AssociatedTyValue<ChalkIr>> {
    let data: AssocTyValue = from_chalk(db, id);
    match data {
        AssocTyValue::TypeAlias(type_alias) => {
            type_alias_associated_ty_value(db, krate, type_alias)
        }
        _ => Arc::new(builtin::associated_ty_value(db, krate, data).to_chalk(db)),
    }
}

fn type_alias_associated_ty_value(
    db: &impl HirDatabase,
    _krate: CrateId,
    type_alias: TypeAliasId,
) -> Arc<AssociatedTyValue<ChalkIr>> {
    let type_alias_data = db.type_alias_data(type_alias);
    let impl_id = match type_alias.lookup(db).container {
        ContainerId::ImplId(it) => it,
        _ => panic!("assoc ty value should be in impl"),
    };

    let trait_ref = db.impl_trait(impl_id).expect("assoc ty value should not exist"); // we don't return any assoc ty values if the impl'd trait can't be resolved

    let assoc_ty = db
        .trait_data(trait_ref.trait_)
        .associated_type_by_name(&type_alias_data.name)
        .expect("assoc ty value should not exist"); // validated when building the impl data as well
    let generic_params = db.generic_params(impl_id.into());
    let bound_vars = Substs::bound_vars(&generic_params);
    let ty = db.ty(type_alias.into()).subst(&bound_vars);
    let value_bound = chalk_rust_ir::AssociatedTyValueBound { ty: ty.to_chalk(db) };
    let value = chalk_rust_ir::AssociatedTyValue {
        impl_id: Impl::ImplBlock(impl_id.into()).to_chalk(db),
        associated_ty_id: assoc_ty.to_chalk(db),
        value: make_binders(value_bound, bound_vars.len()),
    };
    Arc::new(value)
}

fn id_from_chalk<T: InternKey>(chalk_id: chalk_ir::RawId) -> T {
    T::from_intern_id(InternId::from(chalk_id.index))
}
fn id_to_chalk<T: InternKey>(salsa_id: T) -> chalk_ir::RawId {
    chalk_ir::RawId { index: salsa_id.as_intern_id().as_u32() }
}

impl From<chalk_ir::StructId> for crate::TypeCtorId {
    fn from(struct_id: chalk_ir::StructId) -> Self {
        id_from_chalk(struct_id.0)
    }
}

impl From<crate::TypeCtorId> for chalk_ir::StructId {
    fn from(type_ctor_id: crate::TypeCtorId) -> Self {
        chalk_ir::StructId(id_to_chalk(type_ctor_id))
    }
}

impl From<chalk_ir::ImplId> for crate::traits::GlobalImplId {
    fn from(impl_id: chalk_ir::ImplId) -> Self {
        id_from_chalk(impl_id.0)
    }
}

impl From<crate::traits::GlobalImplId> for chalk_ir::ImplId {
    fn from(impl_id: crate::traits::GlobalImplId) -> Self {
        chalk_ir::ImplId(id_to_chalk(impl_id))
    }
}

impl From<chalk_rust_ir::AssociatedTyValueId> for crate::traits::AssocTyValueId {
    fn from(id: chalk_rust_ir::AssociatedTyValueId) -> Self {
        id_from_chalk(id.0)
    }
}

impl From<crate::traits::AssocTyValueId> for chalk_rust_ir::AssociatedTyValueId {
    fn from(assoc_ty_value_id: crate::traits::AssocTyValueId) -> Self {
        chalk_rust_ir::AssociatedTyValueId(id_to_chalk(assoc_ty_value_id))
    }
}
