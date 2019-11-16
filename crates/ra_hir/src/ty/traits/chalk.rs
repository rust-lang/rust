//! Conversion code from/to Chalk.
use std::sync::Arc;

use log::debug;

use chalk_ir::{
    cast::Cast, family::ChalkIr, Identifier, ImplId, Parameter, PlaceholderIndex, TypeId,
    TypeKindId, TypeName, UniverseIndex,
};
use chalk_rust_ir::{AssociatedTyDatum, AssociatedTyValue, ImplDatum, StructDatum, TraitDatum};

use hir_expand::name;

use ra_db::salsa::{InternId, InternKey};

use super::{AssocTyValue, Canonical, ChalkContext, Impl, Obligation};
use crate::{
    db::HirDatabase,
    generics::{GenericDef, HasGenericParams},
    ty::display::HirDisplay,
    ty::{ApplicationTy, GenericPredicate, ProjectionTy, Substs, TraitRef, Ty, TypeCtor, TypeWalk},
    Crate, HasBody, ImplBlock, Trait, TypeAlias,
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

impl ToChalk for Trait {
    type Chalk = chalk_ir::TraitId;

    fn to_chalk(self, _db: &impl HirDatabase) -> chalk_ir::TraitId {
        chalk_ir::TraitId(id_to_chalk(self.id))
    }

    fn from_chalk(_db: &impl HirDatabase, trait_id: chalk_ir::TraitId) -> Trait {
        Trait { id: id_from_chalk(trait_id.0) }
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

impl ToChalk for TypeAlias {
    type Chalk = chalk_ir::TypeId;

    fn to_chalk(self, _db: &impl HirDatabase) -> chalk_ir::TypeId {
        chalk_ir::TypeId(id_to_chalk(self.id))
    }

    fn from_chalk(_db: &impl HirDatabase, type_alias_id: chalk_ir::TypeId) -> TypeAlias {
        TypeAlias { id: id_from_chalk(type_alias_id.0) }
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

fn make_binders<T>(value: T, num_vars: usize) -> chalk_ir::Binders<T> {
    chalk_ir::Binders {
        value,
        binders: std::iter::repeat(chalk_ir::ParameterKind::Ty(())).take(num_vars).collect(),
    }
}

fn convert_where_clauses(
    db: &impl HirDatabase,
    def: GenericDef,
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
    fn impl_datum(&self, impl_id: ImplId) -> Arc<ImplDatum<ChalkIr>> {
        self.db.impl_datum(self.krate, impl_id)
    }
    fn impls_for_trait(
        &self,
        trait_id: chalk_ir::TraitId,
        parameters: &[Parameter<ChalkIr>],
    ) -> Vec<ImplId> {
        debug!("impls_for_trait {:?}", trait_id);
        if trait_id == UNKNOWN_TRAIT {
            return Vec::new();
        }
        let trait_: Trait = from_chalk(self.db, trait_id);
        let mut result: Vec<_> = self
            .db
            .impls_for_trait(self.krate, trait_)
            .iter()
            .copied()
            .map(Impl::ImplBlock)
            .map(|impl_| impl_.to_chalk(self.db))
            .collect();

        let ty: Ty = from_chalk(self.db, parameters[0].assert_ty_ref().clone());
        if let Ty::Apply(ApplicationTy { ctor: TypeCtor::Closure { def, expr }, .. }) = ty {
            for &fn_trait in
                [super::FnTrait::FnOnce, super::FnTrait::FnMut, super::FnTrait::Fn].iter()
            {
                if let Some(actual_trait) = get_fn_trait(self.db, self.krate, fn_trait) {
                    if trait_ == actual_trait {
                        let impl_ = super::ClosureFnTraitImplData { def, expr, fn_trait };
                        result.push(Impl::ClosureFnTraitImpl(impl_).to_chalk(self.db));
                    }
                }
            }
        }

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
        self.db.associated_ty_value(self.krate, id)
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
    let type_alias: TypeAlias = from_chalk(db, id);
    let trait_ = match type_alias.container(db) {
        Some(crate::Container::Trait(t)) => t,
        _ => panic!("associated type not in trait"),
    };
    let generic_params = type_alias.generic_params(db);
    let bound_data = chalk_rust_ir::AssociatedTyDatumBound {
        // FIXME add bounds and where clauses
        bounds: vec![],
        where_clauses: vec![],
    };
    let datum = AssociatedTyDatum {
        trait_id: trait_.to_chalk(db),
        id,
        name: lalrpop_intern::intern(&type_alias.name(db).to_string()),
        binders: make_binders(bound_data, generic_params.count_params_including_parent()),
    };
    Arc::new(datum)
}

pub(crate) fn trait_datum_query(
    db: &impl HirDatabase,
    krate: Crate,
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
    let trait_: Trait = from_chalk(db, trait_id);
    debug!("trait {:?} = {:?}", trait_id, trait_.name(db));
    let generic_params = trait_.generic_params(db);
    let bound_vars = Substs::bound_vars(&generic_params);
    let flags = chalk_rust_ir::TraitFlags {
        auto: trait_.is_auto(db),
        upstream: trait_.module(db).krate() != krate,
        non_enumerable: true,
        coinductive: false, // only relevant for Chalk testing
        // FIXME set these flags correctly
        marker: false,
        fundamental: false,
    };
    let where_clauses = convert_where_clauses(db, trait_.into(), &bound_vars);
    let associated_ty_ids = trait_
        .items(db)
        .into_iter()
        .filter_map(|trait_item| match trait_item {
            crate::AssocItem::TypeAlias(type_alias) => Some(type_alias),
            _ => None,
        })
        .map(|type_alias| type_alias.to_chalk(db))
        .collect();
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
    krate: Crate,
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
            let generic_params = generic_def.generic_params(db);
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
    krate: Crate,
    impl_id: ImplId,
) -> Arc<ImplDatum<ChalkIr>> {
    let _p = ra_prof::profile("impl_datum");
    debug!("impl_datum {:?}", impl_id);
    let impl_: Impl = from_chalk(db, impl_id);
    match impl_ {
        Impl::ImplBlock(impl_block) => impl_block_datum(db, krate, impl_id, impl_block),
        Impl::ClosureFnTraitImpl(data) => closure_fn_trait_impl_datum(db, krate, data),
    }
    .unwrap_or_else(invalid_impl_datum)
}

fn impl_block_datum(
    db: &impl HirDatabase,
    krate: Crate,
    impl_id: ImplId,
    impl_block: ImplBlock,
) -> Option<Arc<ImplDatum<ChalkIr>>> {
    let generic_params = impl_block.generic_params(db);
    let bound_vars = Substs::bound_vars(&generic_params);
    let trait_ref = impl_block.target_trait_ref(db)?.subst(&bound_vars);
    let trait_ = trait_ref.trait_;
    let impl_type = if impl_block.krate(db) == krate {
        chalk_rust_ir::ImplType::Local
    } else {
        chalk_rust_ir::ImplType::External
    };
    let where_clauses = convert_where_clauses(db, impl_block.into(), &bound_vars);
    let negative = impl_block.is_negative(db);
    debug!(
        "impl {:?}: {}{} where {:?}",
        impl_id,
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
    let associated_ty_value_ids = impl_block
        .items(db)
        .into_iter()
        .filter_map(|item| match item {
            crate::AssocItem::TypeAlias(type_alias) => Some(type_alias),
            _ => None,
        })
        .filter(|type_alias| {
            // don't include associated types that don't exist in the trait
            trait_.associated_type_by_name(db, &type_alias.name(db)).is_some()
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

fn closure_fn_trait_impl_datum(
    db: &impl HirDatabase,
    krate: Crate,
    data: super::ClosureFnTraitImplData,
) -> Option<Arc<ImplDatum<ChalkIr>>> {
    // for some closure |X, Y| -> Z:
    // impl<T, U, V> Fn<(T, U)> for closure<fn(T, U) -> V> { Output = V }

    let trait_ = get_fn_trait(db, krate, data.fn_trait)?; // get corresponding fn trait

    // validate FnOnce trait, since we need it in the assoc ty value definition
    // and don't want to return a valid value only to find out later that FnOnce
    // is broken
    let fn_once_trait = get_fn_trait(db, krate, super::FnTrait::FnOnce)?;
    fn_once_trait.associated_type_by_name(db, &name::OUTPUT_TYPE)?;

    let num_args: u16 = match &data.def.body(db)[data.expr] {
        crate::expr::Expr::Lambda { args, .. } => args.len() as u16,
        _ => {
            log::warn!("closure for closure type {:?} not found", data);
            0
        }
    };

    let arg_ty = Ty::apply(
        TypeCtor::Tuple { cardinality: num_args },
        Substs::builder(num_args as usize).fill_with_bound_vars(0).build(),
    );
    let sig_ty = Ty::apply(
        TypeCtor::FnPtr { num_args },
        Substs::builder(num_args as usize + 1).fill_with_bound_vars(0).build(),
    );

    let self_ty = Ty::apply_one(TypeCtor::Closure { def: data.def, expr: data.expr }, sig_ty);

    let trait_ref = TraitRef {
        trait_,
        substs: Substs::build_for_def(db, trait_).push(self_ty).push(arg_ty).build(),
    };

    let output_ty_id = AssocTyValue::ClosureFnTraitImplOutput(data.clone()).to_chalk(db);

    let impl_type = chalk_rust_ir::ImplType::External;

    let impl_datum_bound = chalk_rust_ir::ImplDatumBound {
        trait_ref: trait_ref.to_chalk(db),
        where_clauses: Vec::new(),
    };
    let impl_datum = ImplDatum {
        binders: make_binders(impl_datum_bound, num_args as usize + 1),
        impl_type,
        polarity: chalk_rust_ir::Polarity::Positive,
        associated_ty_value_ids: vec![output_ty_id],
    };
    Some(Arc::new(impl_datum))
}

pub(crate) fn associated_ty_value_query(
    db: &impl HirDatabase,
    krate: Crate,
    id: chalk_rust_ir::AssociatedTyValueId,
) -> Arc<chalk_rust_ir::AssociatedTyValue<ChalkIr>> {
    let data: AssocTyValue = from_chalk(db, id);
    match data {
        AssocTyValue::TypeAlias(type_alias) => {
            type_alias_associated_ty_value(db, krate, type_alias)
        }
        AssocTyValue::ClosureFnTraitImplOutput(data) => {
            closure_fn_trait_output_assoc_ty_value(db, krate, data)
        }
    }
}

fn type_alias_associated_ty_value(
    db: &impl HirDatabase,
    _krate: Crate,
    type_alias: TypeAlias,
) -> Arc<AssociatedTyValue<ChalkIr>> {
    let impl_block = type_alias.impl_block(db).expect("assoc ty value should be in impl");
    let impl_id = Impl::ImplBlock(impl_block).to_chalk(db);
    let trait_ = impl_block
        .target_trait_ref(db)
        .expect("assoc ty value should not exist") // we don't return any assoc ty values if the impl'd trait can't be resolved
        .trait_;
    let assoc_ty = trait_
        .associated_type_by_name(db, &type_alias.name(db))
        .expect("assoc ty value should not exist"); // validated when building the impl data as well
    let generic_params = impl_block.generic_params(db);
    let bound_vars = Substs::bound_vars(&generic_params);
    let ty = db.type_for_def(type_alias.into(), crate::ty::Namespace::Types).subst(&bound_vars);
    let value_bound = chalk_rust_ir::AssociatedTyValueBound { ty: ty.to_chalk(db) };
    let value = chalk_rust_ir::AssociatedTyValue {
        impl_id,
        associated_ty_id: assoc_ty.to_chalk(db),
        value: make_binders(value_bound, bound_vars.len()),
    };
    Arc::new(value)
}

fn closure_fn_trait_output_assoc_ty_value(
    db: &impl HirDatabase,
    krate: Crate,
    data: super::ClosureFnTraitImplData,
) -> Arc<AssociatedTyValue<ChalkIr>> {
    let impl_id = Impl::ClosureFnTraitImpl(data.clone()).to_chalk(db);

    let num_args: u16 = match &data.def.body(db)[data.expr] {
        crate::expr::Expr::Lambda { args, .. } => args.len() as u16,
        _ => {
            log::warn!("closure for closure type {:?} not found", data);
            0
        }
    };

    let output_ty = Ty::Bound(num_args.into());

    let fn_once_trait =
        get_fn_trait(db, krate, super::FnTrait::FnOnce).expect("assoc ty value should not exist");

    let output_ty_id = fn_once_trait
        .associated_type_by_name(db, &name::OUTPUT_TYPE)
        .expect("assoc ty value should not exist");

    let value_bound = chalk_rust_ir::AssociatedTyValueBound { ty: output_ty.to_chalk(db) };

    let value = chalk_rust_ir::AssociatedTyValue {
        associated_ty_id: output_ty_id.to_chalk(db),
        impl_id,
        value: make_binders(value_bound, num_args as usize + 1),
    };
    Arc::new(value)
}

fn get_fn_trait(db: &impl HirDatabase, krate: Crate, fn_trait: super::FnTrait) -> Option<Trait> {
    let target = db.lang_item(krate, fn_trait.lang_item_name().into())?;
    match target {
        crate::lang_item::LangItemTarget::Trait(t) => Some(t),
        _ => None,
    }
}

fn id_from_chalk<T: InternKey>(chalk_id: chalk_ir::RawId) -> T {
    T::from_intern_id(InternId::from(chalk_id.index))
}
fn id_to_chalk<T: InternKey>(salsa_id: T) -> chalk_ir::RawId {
    chalk_ir::RawId { index: salsa_id.as_intern_id().as_u32() }
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

impl From<chalk_rust_ir::AssociatedTyValueId> for crate::ids::AssocTyValueId {
    fn from(id: chalk_rust_ir::AssociatedTyValueId) -> Self {
        id_from_chalk(id.0)
    }
}

impl From<crate::ids::AssocTyValueId> for chalk_rust_ir::AssociatedTyValueId {
    fn from(assoc_ty_value_id: crate::ids::AssocTyValueId) -> Self {
        chalk_rust_ir::AssociatedTyValueId(id_to_chalk(assoc_ty_value_id))
    }
}
