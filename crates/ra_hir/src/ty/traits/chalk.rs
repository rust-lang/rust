//! Conversion code from/to Chalk.
use std::sync::Arc;

use log::debug;

use chalk_ir::{
    cast::Cast, Identifier, ImplId, Parameter, PlaceholderIndex, TypeId, TypeKindId, TypeName,
    UniverseIndex,
};
use chalk_rust_ir::{AssociatedTyDatum, ImplDatum, StructDatum, TraitDatum};

use ra_db::salsa::{InternId, InternKey};
use test_utils::tested_by;

use super::{Canonical, ChalkContext, Impl, Obligation};
use crate::{
    db::HirDatabase,
    generics::GenericDef,
    ty::display::HirDisplay,
    ty::{
        ApplicationTy, CallableDef, GenericPredicate, ProjectionTy, Substs, TraitRef, Ty, TypeCtor,
        TypeWalk,
    },
    AssocItem, Crate, HasGenericParams, ImplBlock, Trait, TypeAlias,
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
    type Chalk = chalk_ir::Ty;
    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::Ty {
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
                chalk_ir::ApplicationTy { name, parameters }.cast()
            }
            Ty::Projection(proj_ty) => {
                let associated_ty_id = proj_ty.associated_ty.to_chalk(db);
                let parameters = proj_ty.parameters.to_chalk(db);
                chalk_ir::ProjectionTy { associated_ty_id, parameters }.cast()
            }
            Ty::Param { idx, .. } => {
                PlaceholderIndex { ui: UniverseIndex::ROOT, idx: idx as usize }.to_ty()
            }
            Ty::Bound(idx) => chalk_ir::Ty::BoundVar(idx as usize),
            Ty::Infer(_infer_ty) => panic!("uncanonicalized infer ty"),
            // FIXME this is clearly incorrect, but probably not too incorrect
            // and I'm not sure what to actually do with Ty::Unknown
            // maybe an alternative would be `for<T> T`? (meaningless in rust, but expressible in chalk's Ty)
            //
            // FIXME also dyn and impl Trait are currently handled like Unknown because Chalk doesn't have them yet
            Ty::Unknown | Ty::Dyn(_) | Ty::Opaque(_) => {
                PlaceholderIndex { ui: UniverseIndex::ROOT, idx: usize::max_value() }.to_ty()
            }
        }
    }
    fn from_chalk(db: &impl HirDatabase, chalk: chalk_ir::Ty) -> Self {
        match chalk {
            chalk_ir::Ty::Apply(apply_ty) => {
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
                    // FIXME handle TypeKindId::Trait/Type here
                    TypeName::TypeKindId(_) => unimplemented!(),
                    TypeName::Placeholder(idx) => {
                        assert_eq!(idx.ui, UniverseIndex::ROOT);
                        Ty::Param { idx: idx.idx as u32, name: crate::Name::missing() }
                    }
                }
            }
            chalk_ir::Ty::Projection(proj) => {
                let associated_ty = from_chalk(db, proj.associated_ty_id);
                let parameters = from_chalk(db, proj.parameters);
                Ty::Projection(ProjectionTy { associated_ty, parameters })
            }
            chalk_ir::Ty::ForAll(_) => unimplemented!(),
            chalk_ir::Ty::BoundVar(idx) => Ty::Bound(idx as u32),
            chalk_ir::Ty::InferenceVar(_iv) => Ty::Unknown,
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

impl ToChalk for Impl {
    type Chalk = chalk_ir::ImplId;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::ImplId {
        db.intern_impl(self).into()
    }

    fn from_chalk(db: &impl HirDatabase, impl_id: chalk_ir::ImplId) -> Impl {
        db.lookup_intern_impl(impl_id.into())
    }
}

impl ToChalk for TypeAlias {
    type Chalk = chalk_ir::TypeId;

    fn to_chalk(self, _db: &impl HirDatabase) -> chalk_ir::TypeId {
        self.id.into()
    }

    fn from_chalk(_db: &impl HirDatabase, impl_id: chalk_ir::TypeId) -> TypeAlias {
        TypeAlias { id: impl_id.into() }
    }
}

impl ToChalk for GenericPredicate {
    type Chalk = chalk_ir::QuantifiedWhereClause;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::QuantifiedWhereClause {
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
        _db: &impl HirDatabase,
        _where_clause: chalk_ir::QuantifiedWhereClause,
    ) -> GenericPredicate {
        // This should never need to be called
        unimplemented!()
    }
}

impl ToChalk for ProjectionTy {
    type Chalk = chalk_ir::ProjectionTy;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::ProjectionTy {
        chalk_ir::ProjectionTy {
            associated_ty_id: self.associated_ty.to_chalk(db),
            parameters: self.parameters.to_chalk(db),
        }
    }

    fn from_chalk(db: &impl HirDatabase, projection_ty: chalk_ir::ProjectionTy) -> ProjectionTy {
        ProjectionTy {
            associated_ty: from_chalk(db, projection_ty.associated_ty_id),
            parameters: from_chalk(db, projection_ty.parameters),
        }
    }
}

impl ToChalk for super::ProjectionPredicate {
    type Chalk = chalk_ir::Normalize;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::Normalize {
        chalk_ir::Normalize {
            projection: self.projection_ty.to_chalk(db),
            ty: self.ty.to_chalk(db),
        }
    }

    fn from_chalk(_db: &impl HirDatabase, _normalize: chalk_ir::Normalize) -> Self {
        unimplemented!()
    }
}

impl ToChalk for Obligation {
    type Chalk = chalk_ir::DomainGoal;

    fn to_chalk(self, db: &impl HirDatabase) -> chalk_ir::DomainGoal {
        match self {
            Obligation::Trait(tr) => tr.to_chalk(db).cast(),
            Obligation::Projection(pr) => pr.to_chalk(db).cast(),
        }
    }

    fn from_chalk(_db: &impl HirDatabase, _goal: chalk_ir::DomainGoal) -> Self {
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
    type Chalk = Arc<chalk_ir::Environment>;

    fn to_chalk(self, db: &impl HirDatabase) -> Arc<chalk_ir::Environment> {
        let mut clauses = Vec::new();
        for pred in &self.predicates {
            if pred.is_error() {
                // for env, we just ignore errors
                continue;
            }
            let program_clause: chalk_ir::ProgramClause = pred.clone().to_chalk(db).cast();
            clauses.push(program_clause.into_from_env_clause());
        }
        chalk_ir::Environment::new().add_clauses(clauses)
    }

    fn from_chalk(
        _db: &impl HirDatabase,
        _env: Arc<chalk_ir::Environment>,
    ) -> Arc<super::TraitEnvironment> {
        unimplemented!()
    }
}

impl<T: ToChalk> ToChalk for super::InEnvironment<T> {
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
) -> Vec<chalk_ir::QuantifiedWhereClause> {
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

impl<'a, DB> chalk_solve::RustIrDatabase for ChalkContext<'a, DB>
where
    DB: HirDatabase,
{
    fn associated_ty_data(&self, id: TypeId) -> Arc<AssociatedTyDatum> {
        self.db.associated_ty_data(id)
    }
    fn trait_datum(&self, trait_id: chalk_ir::TraitId) -> Arc<TraitDatum> {
        self.db.trait_datum(self.krate, trait_id)
    }
    fn struct_datum(&self, struct_id: chalk_ir::StructId) -> Arc<StructDatum> {
        self.db.struct_datum(self.krate, struct_id)
    }
    fn impl_datum(&self, impl_id: ImplId) -> Arc<ImplDatum> {
        self.db.impl_datum(self.krate, impl_id)
    }
    fn impls_for_trait(
        &self,
        trait_id: chalk_ir::TraitId,
        parameters: &[Parameter],
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
            for fn_trait in
                [super::FnTrait::FnOnce, super::FnTrait::FnMut, super::FnTrait::Fn].iter().copied()
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
    fn split_projection<'p>(
        &self,
        projection: &'p chalk_ir::ProjectionTy,
    ) -> (Arc<AssociatedTyDatum>, &'p [Parameter], &'p [Parameter]) {
        debug!("split_projection {:?}", projection);
        // we don't support GATs, so I think this should always be correct currently
        (self.db.associated_ty_data(projection.associated_ty_id), &projection.parameters, &[])
    }
    fn custom_clauses(&self) -> Vec<chalk_ir::ProgramClause> {
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
) -> Arc<AssociatedTyDatum> {
    debug!("associated_ty_data {:?}", id);
    let type_alias: TypeAlias = from_chalk(db, id);
    let trait_ = match type_alias.container(db) {
        Some(crate::Container::Trait(t)) => t,
        _ => panic!("associated type not in trait"),
    };
    let generic_params = type_alias.generic_params(db);
    let parameter_kinds = generic_params
        .params_including_parent()
        .into_iter()
        .map(|p| chalk_ir::ParameterKind::Ty(lalrpop_intern::intern(&p.name.to_string())))
        .collect();
    let datum = AssociatedTyDatum {
        trait_id: trait_.to_chalk(db),
        id,
        name: lalrpop_intern::intern(&type_alias.name(db).to_string()),
        parameter_kinds,
        // FIXME add bounds and where clauses
        bounds: vec![],
        where_clauses: vec![],
    };
    Arc::new(datum)
}

pub(crate) fn trait_datum_query(
    db: &impl HirDatabase,
    krate: Crate,
    trait_id: chalk_ir::TraitId,
) -> Arc<TraitDatum> {
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
                non_enumerable: true,
                auto: false,
                marker: false,
                upstream: true,
                fundamental: false,
            },
        };
        return Arc::new(TraitDatum { binders: make_binders(trait_datum_bound, 1) });
    }
    let trait_: Trait = from_chalk(db, trait_id);
    debug!("trait {:?} = {:?}", trait_id, trait_.name(db));
    let generic_params = trait_.generic_params(db);
    let bound_vars = Substs::bound_vars(&generic_params);
    let trait_ref = trait_.trait_ref(db).subst(&bound_vars).to_chalk(db);
    let flags = chalk_rust_ir::TraitFlags {
        auto: trait_.is_auto(db),
        upstream: trait_.module(db).krate(db) != Some(krate),
        non_enumerable: true,
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
    let trait_datum_bound =
        chalk_rust_ir::TraitDatumBound { trait_ref, where_clauses, flags, associated_ty_ids };
    let trait_datum = TraitDatum { binders: make_binders(trait_datum_bound, bound_vars.len()) };
    Arc::new(trait_datum)
}

pub(crate) fn struct_datum_query(
    db: &impl HirDatabase,
    krate: Crate,
    struct_id: chalk_ir::StructId,
) -> Arc<StructDatum> {
    debug!("struct_datum {:?}", struct_id);
    let type_ctor = from_chalk(db, struct_id);
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
            let upstream = match callable {
                CallableDef::Function(f) => f.module(db).krate(db),
                CallableDef::Struct(s) => s.module(db).krate(db),
                CallableDef::EnumVariant(v) => v.parent_enum(db).module(db).krate(db),
            } != Some(krate);
            let generic_def: GenericDef = callable.into();
            let generic_params = generic_def.generic_params(db);
            let bound_vars = Substs::bound_vars(&generic_params);
            let where_clauses = convert_where_clauses(db, generic_def, &bound_vars);
            (generic_params.count_params_including_parent(), where_clauses, upstream)
        }
        TypeCtor::Adt(adt) => {
            let generic_params = adt.generic_params(db);
            let bound_vars = Substs::bound_vars(&generic_params);
            let where_clauses = convert_where_clauses(db, adt.into(), &bound_vars);
            (
                generic_params.count_params_including_parent(),
                where_clauses,
                adt.krate(db) != Some(krate),
            )
        }
        TypeCtor::AssociatedType(type_alias) => {
            let generic_params = type_alias.generic_params(db);
            let bound_vars = Substs::bound_vars(&generic_params);
            let where_clauses = convert_where_clauses(db, type_alias.into(), &bound_vars);
            (
                generic_params.count_params_including_parent(),
                where_clauses,
                type_alias.krate(db) != Some(krate),
            )
        }
        TypeCtor::Closure { def, .. } => {
            let upstream = def.krate(db) != Some(krate);
            (1, vec![], upstream)
        }
    };
    let flags = chalk_rust_ir::StructFlags {
        upstream,
        // FIXME set fundamental flag correctly
        fundamental: false,
    };
    let self_ty = chalk_ir::ApplicationTy {
        name: TypeName::TypeKindId(type_ctor.to_chalk(db).into()),
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

pub(crate) fn impl_datum_query(
    db: &impl HirDatabase,
    krate: Crate,
    impl_id: ImplId,
) -> Arc<ImplDatum> {
    let _p = ra_prof::profile("impl_datum");
    debug!("impl_datum {:?}", impl_id);
    let impl_: Impl = from_chalk(db, impl_id);
    match impl_ {
        Impl::ImplBlock(impl_block) => impl_block_datum(db, krate, impl_id, impl_block),
        Impl::ClosureFnTraitImpl(data) => {
            closure_fn_trait_impl_datum(db, krate, impl_id, data).unwrap_or_else(invalid_impl_datum)
        }
    }
}

fn impl_block_datum(
    db: &impl HirDatabase,
    krate: Crate,
    impl_id: ImplId,
    impl_block: ImplBlock,
) -> Arc<ImplDatum> {
    let generic_params = impl_block.generic_params(db);
    let bound_vars = Substs::bound_vars(&generic_params);
    let trait_ref = impl_block
        .target_trait_ref(db)
        .expect("FIXME handle unresolved impl block trait ref")
        .subst(&bound_vars);
    let impl_type = if impl_block.module().krate(db) == Some(krate) {
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
    let trait_ = trait_ref.trait_;
    let trait_ref = trait_ref.to_chalk(db);
    let associated_ty_values = impl_block
        .items(db)
        .into_iter()
        .filter_map(|item| match item {
            AssocItem::TypeAlias(t) => Some(t),
            _ => None,
        })
        .filter_map(|t| {
            let assoc_ty = trait_.associated_type_by_name(db, &t.name(db))?;
            let ty = db.type_for_def(t.into(), crate::Namespace::Types).subst(&bound_vars);
            Some(chalk_rust_ir::AssociatedTyValue {
                impl_id,
                associated_ty_id: assoc_ty.to_chalk(db),
                value: chalk_ir::Binders {
                    value: chalk_rust_ir::AssociatedTyValueBound { ty: ty.to_chalk(db) },
                    binders: vec![], // we don't support GATs yet
                },
            })
        })
        .collect();

    let impl_datum_bound = chalk_rust_ir::ImplDatumBound {
        trait_ref: if negative {
            chalk_rust_ir::PolarizedTraitRef::Negative(trait_ref)
        } else {
            chalk_rust_ir::PolarizedTraitRef::Positive(trait_ref)
        },
        where_clauses,
        associated_ty_values,
        impl_type,
    };
    debug!("impl_datum: {:?}", impl_datum_bound);
    let impl_datum = ImplDatum { binders: make_binders(impl_datum_bound, bound_vars.len()) };
    Arc::new(impl_datum)
}

fn invalid_impl_datum() -> Arc<ImplDatum> {
    let trait_ref = chalk_ir::TraitRef {
        trait_id: UNKNOWN_TRAIT,
        parameters: vec![chalk_ir::Ty::BoundVar(0).cast()],
    };
    let impl_datum_bound = chalk_rust_ir::ImplDatumBound {
        trait_ref: chalk_rust_ir::PolarizedTraitRef::Positive(trait_ref),
        where_clauses: Vec::new(),
        associated_ty_values: Vec::new(),
        impl_type: chalk_rust_ir::ImplType::External,
    };
    let impl_datum = ImplDatum { binders: make_binders(impl_datum_bound, 1) };
    Arc::new(impl_datum)
}

fn closure_fn_trait_impl_datum(
    db: &impl HirDatabase,
    krate: Crate,
    impl_id: ImplId,
    data: super::ClosureFnTraitImplData,
) -> Option<Arc<ImplDatum>> {
    // for some closure |X, Y| -> Z:
    // impl<T, U, V> Fn<(T, U)> for closure<fn(T, U) -> V> { Output = V }

    let fn_once_trait = get_fn_trait(db, krate, super::FnTrait::FnOnce)?;
    let trait_ = get_fn_trait(db, krate, data.fn_trait)?; // get corresponding fn trait

    let num_args: u16 = match &db.body_hir(data.def)[data.expr] {
        crate::expr::Expr::Lambda { args, .. } => args.len() as u16,
        _ => {
            log::warn!("closure for closure type {:?} not found", data);
            0
        }
    };

    let arg_ty = Ty::apply(
        TypeCtor::Tuple { cardinality: num_args },
        (0..num_args).map(|i| Ty::Bound(i.into())).collect::<Vec<_>>().into(),
    );
    let output_ty = Ty::Bound(num_args.into());
    let sig_ty = Ty::apply(
        TypeCtor::FnPtr { num_args },
        (0..num_args + 1).map(|i| Ty::Bound(i.into())).collect::<Vec<_>>().into(),
    );

    let self_ty = Ty::apply_one(TypeCtor::Closure { def: data.def, expr: data.expr }, sig_ty);

    let trait_ref = TraitRef { trait_, substs: vec![self_ty, arg_ty].into() };

    let output_ty_id = fn_once_trait.associated_type_by_name(db, &crate::name::OUTPUT_TYPE)?;

    let output_ty_value = chalk_rust_ir::AssociatedTyValue {
        associated_ty_id: output_ty_id.to_chalk(db),
        impl_id,
        value: make_binders(
            chalk_rust_ir::AssociatedTyValueBound { ty: output_ty.to_chalk(db) },
            0,
        ),
    };

    let impl_type = chalk_rust_ir::ImplType::External;

    let impl_datum_bound = chalk_rust_ir::ImplDatumBound {
        trait_ref: chalk_rust_ir::PolarizedTraitRef::Positive(trait_ref.to_chalk(db)),
        where_clauses: Vec::new(),
        associated_ty_values: vec![output_ty_value],
        impl_type,
    };
    let impl_datum = ImplDatum { binders: make_binders(impl_datum_bound, num_args as usize + 1) };
    Some(Arc::new(impl_datum))
}

fn get_fn_trait(db: &impl HirDatabase, krate: Crate, fn_trait: super::FnTrait) -> Option<Trait> {
    let lang_items = db.lang_items(krate);
    let target = lang_items.target(fn_trait.lang_item_name())?;
    match target {
        crate::lang_item::LangItemTarget::Trait(t) => Some(*t),
        _ => None,
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

impl From<chalk_ir::TypeId> for crate::ids::TypeAliasId {
    fn from(type_id: chalk_ir::TypeId) -> Self {
        id_from_chalk(type_id.0)
    }
}

impl From<crate::ids::TypeAliasId> for chalk_ir::TypeId {
    fn from(type_id: crate::ids::TypeAliasId) -> Self {
        chalk_ir::TypeId(id_to_chalk(type_id))
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
