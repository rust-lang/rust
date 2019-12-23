//! Conversion code from/to Chalk.
use std::{fmt, sync::Arc};

use log::debug;

use chalk_ir::{cast::Cast, Parameter, PlaceholderIndex, TypeName, UniverseIndex};

use hir_def::{AssocContainerId, AssocItemId, GenericDefId, HasModule, Lookup, TypeAliasId};
use ra_db::{
    salsa::{InternId, InternKey},
    CrateId,
};

use super::{builtin, AssocTyValue, Canonical, ChalkContext, Impl, Obligation};
use crate::{
    db::HirDatabase, display::HirDisplay, utils::generics, ApplicationTy, GenericPredicate,
    ProjectionTy, Substs, TraitRef, Ty, TypeCtor, TypeWalk,
};

#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct TypeFamily {}

impl chalk_ir::family::TypeFamily for TypeFamily {
    type InternedType = Box<chalk_ir::TyData<Self>>;
    type InternedLifetime = chalk_ir::LifetimeData<Self>;
    type InternedParameter = chalk_ir::ParameterData<Self>;
    type DefId = InternId;

    // FIXME: implement these
    fn debug_struct_id(
        _type_kind_id: chalk_ir::StructId<Self>,
        _fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        None
    }

    fn debug_trait_id(
        _type_kind_id: chalk_ir::TraitId<Self>,
        _fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        None
    }

    fn debug_assoc_type_id(
        _id: chalk_ir::AssocTypeId<Self>,
        _fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        None
    }

    fn debug_projection(
        _projection: &chalk_ir::ProjectionTy<Self>,
        _fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        None
    }

    fn intern_ty(ty: chalk_ir::TyData<Self>) -> Box<chalk_ir::TyData<Self>> {
        Box::new(ty)
    }

    fn ty_data(ty: &Box<chalk_ir::TyData<Self>>) -> &chalk_ir::TyData<Self> {
        ty
    }

    fn intern_lifetime(lifetime: chalk_ir::LifetimeData<Self>) -> chalk_ir::LifetimeData<Self> {
        lifetime
    }

    fn lifetime_data(lifetime: &chalk_ir::LifetimeData<Self>) -> &chalk_ir::LifetimeData<Self> {
        lifetime
    }

    fn intern_parameter(parameter: chalk_ir::ParameterData<Self>) -> chalk_ir::ParameterData<Self> {
        parameter
    }

    fn parameter_data(parameter: &chalk_ir::ParameterData<Self>) -> &chalk_ir::ParameterData<Self> {
        parameter
    }
}

impl chalk_ir::family::HasTypeFamily for TypeFamily {
    type TypeFamily = Self;
}

pub type AssocTypeId = chalk_ir::AssocTypeId<TypeFamily>;
pub type AssociatedTyDatum = chalk_rust_ir::AssociatedTyDatum<TypeFamily>;
pub type TraitId = chalk_ir::TraitId<TypeFamily>;
pub type TraitDatum = chalk_rust_ir::TraitDatum<TypeFamily>;
pub type StructId = chalk_ir::StructId<TypeFamily>;
pub type StructDatum = chalk_rust_ir::StructDatum<TypeFamily>;
pub type ImplId = chalk_ir::ImplId<TypeFamily>;
pub type ImplDatum = chalk_rust_ir::ImplDatum<TypeFamily>;
pub type AssociatedTyValueId = chalk_rust_ir::AssociatedTyValueId;
pub type AssociatedTyValue = chalk_rust_ir::AssociatedTyValue<TypeFamily>;

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
    type Chalk = chalk_ir::Ty<TypeFamily>;
    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::Ty<TypeFamily> {
        match self {
            Ty::Apply(apply_ty) => {
                let name = apply_ty.ctor.to_chalk(db);
                let parameters = apply_ty.parameters.to_chalk(db);
                chalk_ir::ApplicationTy { name, parameters }.cast().intern()
            }
            Ty::Projection(proj_ty) => {
                let associated_ty_id = proj_ty.associated_ty.to_chalk(db);
                let parameters = proj_ty.parameters.to_chalk(db);
                chalk_ir::ProjectionTy { associated_ty_id, parameters }.cast().intern()
            }
            Ty::Param { idx, .. } => {
                PlaceholderIndex { ui: UniverseIndex::ROOT, idx: idx as usize }
                    .to_ty::<TypeFamily>()
            }
            Ty::Bound(idx) => chalk_ir::TyData::BoundVar(idx as usize).intern(),
            Ty::Infer(_infer_ty) => panic!("uncanonicalized infer ty"),
            Ty::Dyn(predicates) => {
                let where_clauses = predicates
                    .iter()
                    .filter(|p| !p.is_error())
                    .cloned()
                    .map(|p| p.to_chalk(db))
                    .collect();
                let bounded_ty = chalk_ir::BoundedTy { bounds: make_binders(where_clauses, 1) };
                chalk_ir::TyData::Dyn(bounded_ty).intern()
            }
            Ty::Opaque(predicates) => {
                let where_clauses = predicates
                    .iter()
                    .filter(|p| !p.is_error())
                    .cloned()
                    .map(|p| p.to_chalk(db))
                    .collect();
                let bounded_ty = chalk_ir::BoundedTy { bounds: make_binders(where_clauses, 1) };
                chalk_ir::TyData::Opaque(bounded_ty).intern()
            }
            Ty::Unknown => {
                let parameters = Vec::new();
                let name = TypeName::Error;
                chalk_ir::ApplicationTy { name, parameters }.cast().intern()
            }
        }
    }
    fn from_chalk(db: &impl HirDatabase, chalk: chalk_ir::Ty<TypeFamily>) -> Self {
        match chalk.data().clone() {
            chalk_ir::TyData::Apply(apply_ty) => match apply_ty.name {
                TypeName::Error => Ty::Unknown,
                _ => {
                    let ctor = from_chalk(db, apply_ty.name);
                    let parameters = from_chalk(db, apply_ty.parameters);
                    Ty::Apply(ApplicationTy { ctor, parameters })
                }
            },
            chalk_ir::TyData::Placeholder(idx) => {
                assert_eq!(idx.ui, UniverseIndex::ROOT);
                Ty::Param { idx: idx.idx as u32, name: crate::Name::missing() }
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
                assert_eq!(where_clauses.bounds.binders.len(), 1);
                let predicates =
                    where_clauses.bounds.value.into_iter().map(|c| from_chalk(db, c)).collect();
                Ty::Dyn(predicates)
            }
            chalk_ir::TyData::Opaque(where_clauses) => {
                assert_eq!(where_clauses.bounds.binders.len(), 1);
                let predicates =
                    where_clauses.bounds.value.into_iter().map(|c| from_chalk(db, c)).collect();
                Ty::Opaque(predicates)
            }
        }
    }
}

impl ToChalk for Substs {
    type Chalk = Vec<chalk_ir::Parameter<TypeFamily>>;

    fn to_chalk(self, db: &impl HirDatabase) -> Vec<Parameter<TypeFamily>> {
        self.iter().map(|ty| ty.clone().to_chalk(db).cast()).collect()
    }

    fn from_chalk(
        db: &impl HirDatabase,
        parameters: Vec<chalk_ir::Parameter<TypeFamily>>,
    ) -> Substs {
        let tys = parameters
            .into_iter()
            .map(|p| match p.ty() {
                Some(ty) => from_chalk(db, ty.clone()),
                None => unimplemented!(),
            })
            .collect();
        Substs(tys)
    }
}

impl ToChalk for TraitRef {
    type Chalk = chalk_ir::TraitRef<TypeFamily>;

    fn to_chalk(self: TraitRef, db: &impl HirDatabase) -> chalk_ir::TraitRef<TypeFamily> {
        let trait_id = self.trait_.to_chalk(db);
        let parameters = self.substs.to_chalk(db);
        chalk_ir::TraitRef { trait_id, parameters }
    }

    fn from_chalk(db: &impl HirDatabase, trait_ref: chalk_ir::TraitRef<TypeFamily>) -> Self {
        let trait_ = from_chalk(db, trait_ref.trait_id);
        let substs = from_chalk(db, trait_ref.parameters);
        TraitRef { trait_, substs }
    }
}

impl ToChalk for hir_def::TraitId {
    type Chalk = TraitId;

    fn to_chalk(self, _db: &impl HirDatabase) -> TraitId {
        chalk_ir::TraitId(self.as_intern_id())
    }

    fn from_chalk(_db: &impl HirDatabase, trait_id: TraitId) -> hir_def::TraitId {
        InternKey::from_intern_id(trait_id.0)
    }
}

impl ToChalk for TypeCtor {
    type Chalk = TypeName<TypeFamily>;

    fn to_chalk(self, db: &impl HirDatabase) -> TypeName<TypeFamily> {
        match self {
            TypeCtor::AssociatedType(type_alias) => {
                let type_id = type_alias.to_chalk(db);
                TypeName::AssociatedType(type_id)
            }
            _ => {
                // other TypeCtors get interned and turned into a chalk StructId
                let struct_id = db.intern_type_ctor(self).into();
                TypeName::Struct(struct_id)
            }
        }
    }

    fn from_chalk(db: &impl HirDatabase, type_name: TypeName<TypeFamily>) -> TypeCtor {
        match type_name {
            TypeName::Struct(struct_id) => db.lookup_intern_type_ctor(struct_id.into()),
            TypeName::AssociatedType(type_id) => TypeCtor::AssociatedType(from_chalk(db, type_id)),
            TypeName::Error => {
                // this should not be reached, since we don't represent TypeName::Error with TypeCtor
                unreachable!()
            }
        }
    }
}

impl ToChalk for Impl {
    type Chalk = ImplId;

    fn to_chalk(self, db: &impl HirDatabase) -> ImplId {
        db.intern_chalk_impl(self).into()
    }

    fn from_chalk(db: &impl HirDatabase, impl_id: ImplId) -> Impl {
        db.lookup_intern_chalk_impl(impl_id.into())
    }
}

impl ToChalk for TypeAliasId {
    type Chalk = AssocTypeId;

    fn to_chalk(self, _db: &impl HirDatabase) -> AssocTypeId {
        chalk_ir::AssocTypeId(self.as_intern_id())
    }

    fn from_chalk(_db: &impl HirDatabase, type_alias_id: AssocTypeId) -> TypeAliasId {
        InternKey::from_intern_id(type_alias_id.0)
    }
}

impl ToChalk for AssocTyValue {
    type Chalk = AssociatedTyValueId;

    fn to_chalk(self, db: &impl HirDatabase) -> AssociatedTyValueId {
        db.intern_assoc_ty_value(self).into()
    }

    fn from_chalk(db: &impl HirDatabase, assoc_ty_value_id: AssociatedTyValueId) -> AssocTyValue {
        db.lookup_intern_assoc_ty_value(assoc_ty_value_id.into())
    }
}

impl ToChalk for GenericPredicate {
    type Chalk = chalk_ir::QuantifiedWhereClause<TypeFamily>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::QuantifiedWhereClause<TypeFamily> {
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
            GenericPredicate::Error => panic!("tried passing GenericPredicate::Error to Chalk"),
        }
    }

    fn from_chalk(
        db: &impl HirDatabase,
        where_clause: chalk_ir::QuantifiedWhereClause<TypeFamily>,
    ) -> GenericPredicate {
        match where_clause.value {
            chalk_ir::WhereClause::Implemented(tr) => {
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
    type Chalk = chalk_ir::ProjectionTy<TypeFamily>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::ProjectionTy<TypeFamily> {
        chalk_ir::ProjectionTy {
            associated_ty_id: self.associated_ty.to_chalk(db),
            parameters: self.parameters.to_chalk(db),
        }
    }

    fn from_chalk(
        db: &impl HirDatabase,
        projection_ty: chalk_ir::ProjectionTy<TypeFamily>,
    ) -> ProjectionTy {
        ProjectionTy {
            associated_ty: from_chalk(db, projection_ty.associated_ty_id),
            parameters: from_chalk(db, projection_ty.parameters),
        }
    }
}

impl ToChalk for super::ProjectionPredicate {
    type Chalk = chalk_ir::Normalize<TypeFamily>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::Normalize<TypeFamily> {
        chalk_ir::Normalize {
            projection: self.projection_ty.to_chalk(db),
            ty: self.ty.to_chalk(db),
        }
    }

    fn from_chalk(_db: &impl HirDatabase, _normalize: chalk_ir::Normalize<TypeFamily>) -> Self {
        unimplemented!()
    }
}

impl ToChalk for Obligation {
    type Chalk = chalk_ir::DomainGoal<TypeFamily>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::DomainGoal<TypeFamily> {
        match self {
            Obligation::Trait(tr) => tr.to_chalk(db).cast(),
            Obligation::Projection(pr) => pr.to_chalk(db).cast(),
        }
    }

    fn from_chalk(_db: &impl HirDatabase, _goal: chalk_ir::DomainGoal<TypeFamily>) -> Self {
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
    type Chalk = chalk_ir::Environment<TypeFamily>;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::Environment<TypeFamily> {
        let mut clauses = Vec::new();
        for pred in &self.predicates {
            if pred.is_error() {
                // for env, we just ignore errors
                continue;
            }
            let program_clause: chalk_ir::ProgramClause<TypeFamily> =
                pred.clone().to_chalk(db).cast();
            clauses.push(program_clause.into_from_env_clause());
        }
        chalk_ir::Environment::new().add_clauses(clauses)
    }

    fn from_chalk(
        _db: &impl HirDatabase,
        _env: chalk_ir::Environment<TypeFamily>,
    ) -> Arc<super::TraitEnvironment> {
        unimplemented!()
    }
}

impl<T: ToChalk> ToChalk for super::InEnvironment<T>
where
    T::Chalk: chalk_ir::family::HasTypeFamily<TypeFamily = TypeFamily>,
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
    type Chalk = ImplDatum;

    fn to_chalk(self, db: &impl HirDatabase) -> ImplDatum {
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

    fn from_chalk(_db: &impl HirDatabase, _data: ImplDatum) -> Self {
        unimplemented!()
    }
}

impl ToChalk for builtin::BuiltinImplAssocTyValueData {
    type Chalk = AssociatedTyValue;

    fn to_chalk(self, db: &impl HirDatabase) -> AssociatedTyValue {
        let value_bound = chalk_rust_ir::AssociatedTyValueBound { ty: self.value.to_chalk(db) };

        chalk_rust_ir::AssociatedTyValue {
            associated_ty_id: self.assoc_ty_id.to_chalk(db),
            impl_id: self.impl_.to_chalk(db),
            value: make_binders(value_bound, self.num_vars),
        }
    }

    fn from_chalk(
        _db: &impl HirDatabase,
        _data: AssociatedTyValue,
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
) -> Vec<chalk_ir::QuantifiedWhereClause<TypeFamily>> {
    let generic_predicates = db.generic_predicates(def);
    let mut result = Vec::with_capacity(generic_predicates.len());
    for pred in generic_predicates.iter() {
        if pred.is_error() {
            // skip errored predicates completely
            continue;
        }
        result.push(pred.clone().subst(substs).to_chalk(db));
    }
    result
}

impl<'a, DB> chalk_solve::RustIrDatabase<TypeFamily> for ChalkContext<'a, DB>
where
    DB: HirDatabase,
{
    fn associated_ty_data(&self, id: AssocTypeId) -> Arc<AssociatedTyDatum> {
        self.db.associated_ty_data(id)
    }
    fn trait_datum(&self, trait_id: TraitId) -> Arc<TraitDatum> {
        self.db.trait_datum(self.krate, trait_id)
    }
    fn struct_datum(&self, struct_id: StructId) -> Arc<StructDatum> {
        self.db.struct_datum(self.krate, struct_id)
    }
    fn impl_datum(&self, impl_id: ImplId) -> Arc<ImplDatum> {
        self.db.impl_datum(self.krate, impl_id)
    }
    fn impls_for_trait(
        &self,
        trait_id: TraitId,
        parameters: &[Parameter<TypeFamily>],
    ) -> Vec<ImplId> {
        debug!("impls_for_trait {:?}", trait_id);
        let trait_: hir_def::TraitId = from_chalk(self.db, trait_id);

        // Note: Since we're using impls_for_trait, only impls where the trait
        // can be resolved should ever reach Chalk. `impl_datum` relies on that
        // and will panic if the trait can't be resolved.
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
    fn impl_provided_for(&self, auto_trait_id: TraitId, struct_id: StructId) -> bool {
        debug!("impl_provided_for {:?}, {:?}", auto_trait_id, struct_id);
        false // FIXME
    }
    fn associated_ty_value(&self, id: AssociatedTyValueId) -> Arc<AssociatedTyValue> {
        self.db.associated_ty_value(self.krate.into(), id)
    }
    fn custom_clauses(&self) -> Vec<chalk_ir::ProgramClause<TypeFamily>> {
        vec![]
    }
    fn local_impls_to_coherence_check(&self, _trait_id: TraitId) -> Vec<ImplId> {
        // We don't do coherence checking (yet)
        unimplemented!()
    }
    fn as_struct_id(&self, id: &TypeName<TypeFamily>) -> Option<StructId> {
        match id {
            TypeName::Struct(struct_id) => Some(*struct_id),
            _ => None,
        }
    }
}

pub(crate) fn associated_ty_data_query(
    db: &impl HirDatabase,
    id: AssocTypeId,
) -> Arc<AssociatedTyDatum> {
    debug!("associated_ty_data {:?}", id);
    let type_alias: TypeAliasId = from_chalk(db, id);
    let trait_ = match type_alias.lookup(db).container {
        AssocContainerId::TraitId(t) => t,
        _ => panic!("associated type not in trait"),
    };
    let generic_params = generics(db, type_alias.into());
    let bound_data = chalk_rust_ir::AssociatedTyDatumBound {
        // FIXME add bounds and where clauses
        bounds: vec![],
        where_clauses: vec![],
    };
    let datum = AssociatedTyDatum {
        trait_id: trait_.to_chalk(db),
        id,
        name: lalrpop_intern::intern(&db.type_alias_data(type_alias).name.to_string()),
        binders: make_binders(bound_data, generic_params.len()),
    };
    Arc::new(datum)
}

pub(crate) fn trait_datum_query(
    db: &impl HirDatabase,
    krate: CrateId,
    trait_id: TraitId,
) -> Arc<TraitDatum> {
    debug!("trait_datum {:?}", trait_id);
    let trait_: hir_def::TraitId = from_chalk(db, trait_id);
    let trait_data = db.trait_data(trait_);
    debug!("trait {:?} = {:?}", trait_id, trait_data.name);
    let generic_params = generics(db, trait_.into());
    let bound_vars = Substs::bound_vars(&generic_params);
    let flags = chalk_rust_ir::TraitFlags {
        auto: trait_data.auto,
        upstream: trait_.lookup(db).container.module(db).krate != krate,
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
    struct_id: StructId,
) -> Arc<StructDatum> {
    debug!("struct_datum {:?}", struct_id);
    let type_ctor: TypeCtor = from_chalk(db, TypeName::Struct(struct_id));
    debug!("struct {:?} = {:?}", struct_id, type_ctor);
    let num_params = type_ctor.num_ty_params(db);
    let upstream = type_ctor.krate(db) != Some(krate);
    let where_clauses = type_ctor
        .as_generic_def()
        .map(|generic_def| {
            let generic_params = generics(db, generic_def.into());
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
    impl_id: ImplId,
) -> Arc<ImplDatum> {
    let _p = ra_prof::profile("impl_datum");
    debug!("impl_datum {:?}", impl_id);
    let impl_: Impl = from_chalk(db, impl_id);
    match impl_ {
        Impl::ImplBlock(impl_block) => impl_block_datum(db, krate, impl_id, impl_block),
        _ => Arc::new(builtin::impl_datum(db, krate, impl_).to_chalk(db)),
    }
}

fn impl_block_datum(
    db: &impl HirDatabase,
    krate: CrateId,
    chalk_id: ImplId,
    impl_id: hir_def::ImplId,
) -> Arc<ImplDatum> {
    let trait_ref = db
        .impl_trait(impl_id)
        // ImplIds for impls where the trait ref can't be resolved should never reach Chalk
        .expect("invalid impl passed to Chalk");
    let impl_data = db.impl_data(impl_id);

    let generic_params = generics(db, impl_id.into());
    let bound_vars = Substs::bound_vars(&generic_params);
    let trait_ref = trait_ref.subst(&bound_vars);
    let trait_ = trait_ref.trait_;
    let impl_type = if impl_id.lookup(db).container.module(db).krate == krate {
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
    Arc::new(impl_datum)
}

pub(crate) fn associated_ty_value_query(
    db: &impl HirDatabase,
    krate: CrateId,
    id: AssociatedTyValueId,
) -> Arc<AssociatedTyValue> {
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
) -> Arc<AssociatedTyValue> {
    let type_alias_data = db.type_alias_data(type_alias);
    let impl_id = match type_alias.lookup(db).container {
        AssocContainerId::ImplId(it) => it,
        _ => panic!("assoc ty value should be in impl"),
    };

    let trait_ref = db.impl_trait(impl_id).expect("assoc ty value should not exist"); // we don't return any assoc ty values if the impl'd trait can't be resolved

    let assoc_ty = db
        .trait_data(trait_ref.trait_)
        .associated_type_by_name(&type_alias_data.name)
        .expect("assoc ty value should not exist"); // validated when building the impl data as well
    let generic_params = generics(db, impl_id.into());
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

impl From<StructId> for crate::TypeCtorId {
    fn from(struct_id: StructId) -> Self {
        InternKey::from_intern_id(struct_id.0)
    }
}

impl From<crate::TypeCtorId> for StructId {
    fn from(type_ctor_id: crate::TypeCtorId) -> Self {
        chalk_ir::StructId(type_ctor_id.as_intern_id())
    }
}

impl From<ImplId> for crate::traits::GlobalImplId {
    fn from(impl_id: ImplId) -> Self {
        InternKey::from_intern_id(impl_id.0)
    }
}

impl From<crate::traits::GlobalImplId> for ImplId {
    fn from(impl_id: crate::traits::GlobalImplId) -> Self {
        chalk_ir::ImplId(impl_id.as_intern_id())
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
