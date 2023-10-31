//! Many kinds of items or constructs can have generic parameters: functions,
//! structs, impls, traits, etc. This module provides a common HIR for these
//! generic parameters. See also the `Generics` type and the `generics_of` query
//! in rustc.

use base_db::FileId;
use either::Either;
use hir_expand::{
    name::{AsName, Name},
    ExpandResult, HirFileId, InFile,
};
use intern::Interned;
use la_arena::{Arena, ArenaMap, Idx};
use once_cell::unsync::Lazy;
use stdx::impl_from;
use syntax::ast::{self, HasGenericParams, HasName, HasTypeBounds};
use triomphe::Arc;

use crate::{
    child_by_source::ChildBySource,
    db::DefDatabase,
    dyn_map::{keys, DynMap},
    expander::Expander,
    lower::LowerCtx,
    nameres::{DefMap, MacroSubNs},
    src::{HasChildSource, HasSource},
    type_ref::{LifetimeRef, TypeBound, TypeRef},
    AdtId, ConstParamId, GenericDefId, HasModule, LifetimeParamId, LocalLifetimeParamId,
    LocalTypeOrConstParamId, Lookup, TypeOrConstParamId, TypeParamId,
};

/// Data about a generic type parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct TypeParamData {
    pub name: Option<Name>,
    pub default: Option<Interned<TypeRef>>,
    pub provenance: TypeParamProvenance,
}

/// Data about a generic lifetime parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct LifetimeParamData {
    pub name: Name,
}

/// Data about a generic const parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ConstParamData {
    pub name: Name,
    pub ty: Interned<TypeRef>,
    pub has_default: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum TypeParamProvenance {
    TypeParamList,
    TraitSelf,
    ArgumentImplTrait,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum TypeOrConstParamData {
    TypeParamData(TypeParamData),
    ConstParamData(ConstParamData),
}

impl TypeOrConstParamData {
    pub fn name(&self) -> Option<&Name> {
        match self {
            TypeOrConstParamData::TypeParamData(it) => it.name.as_ref(),
            TypeOrConstParamData::ConstParamData(it) => Some(&it.name),
        }
    }

    pub fn has_default(&self) -> bool {
        match self {
            TypeOrConstParamData::TypeParamData(it) => it.default.is_some(),
            TypeOrConstParamData::ConstParamData(it) => it.has_default,
        }
    }

    pub fn type_param(&self) -> Option<&TypeParamData> {
        match self {
            TypeOrConstParamData::TypeParamData(it) => Some(it),
            TypeOrConstParamData::ConstParamData(_) => None,
        }
    }

    pub fn const_param(&self) -> Option<&ConstParamData> {
        match self {
            TypeOrConstParamData::TypeParamData(_) => None,
            TypeOrConstParamData::ConstParamData(it) => Some(it),
        }
    }

    pub fn is_trait_self(&self) -> bool {
        match self {
            TypeOrConstParamData::TypeParamData(it) => {
                it.provenance == TypeParamProvenance::TraitSelf
            }
            TypeOrConstParamData::ConstParamData(_) => false,
        }
    }
}

impl_from!(TypeParamData, ConstParamData for TypeOrConstParamData);

/// Data about the generic parameters of a function, struct, impl, etc.
#[derive(Clone, PartialEq, Eq, Debug, Default, Hash)]
pub struct GenericParams {
    pub type_or_consts: Arena<TypeOrConstParamData>,
    pub lifetimes: Arena<LifetimeParamData>,
    pub where_predicates: Vec<WherePredicate>,
}

/// A single predicate from a where clause, i.e. `where Type: Trait`. Combined
/// where clauses like `where T: Foo + Bar` are turned into multiple of these.
/// It might still result in multiple actual predicates though, because of
/// associated type bindings like `Iterator<Item = u32>`.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum WherePredicate {
    TypeBound {
        target: WherePredicateTypeTarget,
        bound: Interned<TypeBound>,
    },
    Lifetime {
        target: LifetimeRef,
        bound: LifetimeRef,
    },
    ForLifetime {
        lifetimes: Box<[Name]>,
        target: WherePredicateTypeTarget,
        bound: Interned<TypeBound>,
    },
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum WherePredicateTypeTarget {
    TypeRef(Interned<TypeRef>),
    /// For desugared where predicates that can directly refer to a type param.
    TypeOrConstParam(LocalTypeOrConstParamId),
}

impl GenericParams {
    /// Iterator of type_or_consts field
    pub fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = (Idx<TypeOrConstParamData>, &TypeOrConstParamData)> {
        self.type_or_consts.iter()
    }

    pub(crate) fn generic_params_query(
        db: &dyn DefDatabase,
        def: GenericDefId,
    ) -> Interned<GenericParams> {
        let _p = profile::span("generic_params_query");
        macro_rules! id_to_generics {
            ($id:ident) => {{
                let id = $id.lookup(db).id;
                let tree = id.item_tree(db);
                let item = &tree[id.value];
                item.generic_params.clone()
            }};
        }

        match def {
            GenericDefId::FunctionId(id) => {
                let loc = id.lookup(db);
                let tree = loc.id.item_tree(db);
                let item = &tree[loc.id.value];

                let mut generic_params = GenericParams::clone(&item.explicit_generic_params);

                let module = loc.container.module(db);
                let func_data = db.function_data(id);

                // Don't create an `Expander` nor call `loc.source(db)` if not needed since this
                // causes a reparse after the `ItemTree` has been created.
                let mut expander = Lazy::new(|| {
                    (module.def_map(db), Expander::new(db, loc.source(db).file_id, module))
                });
                for param in &func_data.params {
                    generic_params.fill_implicit_impl_trait_args(db, &mut expander, param);
                }

                Interned::new(generic_params)
            }
            GenericDefId::AdtId(AdtId::StructId(id)) => id_to_generics!(id),
            GenericDefId::AdtId(AdtId::EnumId(id)) => id_to_generics!(id),
            GenericDefId::AdtId(AdtId::UnionId(id)) => id_to_generics!(id),
            GenericDefId::TraitId(id) => id_to_generics!(id),
            GenericDefId::TraitAliasId(id) => id_to_generics!(id),
            GenericDefId::TypeAliasId(id) => id_to_generics!(id),
            GenericDefId::ImplId(id) => id_to_generics!(id),
            GenericDefId::EnumVariantId(_) | GenericDefId::ConstId(_) => {
                Interned::new(GenericParams::default())
            }
        }
    }

    pub(crate) fn fill(&mut self, lower_ctx: &LowerCtx<'_>, node: &dyn HasGenericParams) {
        if let Some(params) = node.generic_param_list() {
            self.fill_params(lower_ctx, params)
        }
        if let Some(where_clause) = node.where_clause() {
            self.fill_where_predicates(lower_ctx, where_clause);
        }
    }

    pub(crate) fn fill_bounds(
        &mut self,
        lower_ctx: &LowerCtx<'_>,
        type_bounds: Option<ast::TypeBoundList>,
        target: Either<TypeRef, LifetimeRef>,
    ) {
        for bound in type_bounds.iter().flat_map(|type_bound_list| type_bound_list.bounds()) {
            self.add_where_predicate_from_bound(lower_ctx, bound, None, target.clone());
        }
    }

    fn fill_params(&mut self, lower_ctx: &LowerCtx<'_>, params: ast::GenericParamList) {
        for type_or_const_param in params.type_or_const_params() {
            match type_or_const_param {
                ast::TypeOrConstParam::Type(type_param) => {
                    let name = type_param.name().map_or_else(Name::missing, |it| it.as_name());
                    // FIXME: Use `Path::from_src`
                    let default = type_param
                        .default_type()
                        .map(|it| Interned::new(TypeRef::from_ast(lower_ctx, it)));
                    let param = TypeParamData {
                        name: Some(name.clone()),
                        default,
                        provenance: TypeParamProvenance::TypeParamList,
                    };
                    self.type_or_consts.alloc(param.into());
                    let type_ref = TypeRef::Path(name.into());
                    self.fill_bounds(
                        lower_ctx,
                        type_param.type_bound_list(),
                        Either::Left(type_ref),
                    );
                }
                ast::TypeOrConstParam::Const(const_param) => {
                    let name = const_param.name().map_or_else(Name::missing, |it| it.as_name());
                    let ty = const_param
                        .ty()
                        .map_or(TypeRef::Error, |it| TypeRef::from_ast(lower_ctx, it));
                    let param = ConstParamData {
                        name,
                        ty: Interned::new(ty),
                        has_default: const_param.default_val().is_some(),
                    };
                    self.type_or_consts.alloc(param.into());
                }
            }
        }
        for lifetime_param in params.lifetime_params() {
            let name =
                lifetime_param.lifetime().map_or_else(Name::missing, |lt| Name::new_lifetime(&lt));
            let param = LifetimeParamData { name: name.clone() };
            self.lifetimes.alloc(param);
            let lifetime_ref = LifetimeRef::new_name(name);
            self.fill_bounds(
                lower_ctx,
                lifetime_param.type_bound_list(),
                Either::Right(lifetime_ref),
            );
        }
    }

    fn fill_where_predicates(&mut self, lower_ctx: &LowerCtx<'_>, where_clause: ast::WhereClause) {
        for pred in where_clause.predicates() {
            let target = if let Some(type_ref) = pred.ty() {
                Either::Left(TypeRef::from_ast(lower_ctx, type_ref))
            } else if let Some(lifetime) = pred.lifetime() {
                Either::Right(LifetimeRef::new(&lifetime))
            } else {
                continue;
            };

            let lifetimes: Option<Box<_>> = pred.generic_param_list().map(|param_list| {
                // Higher-Ranked Trait Bounds
                param_list
                    .lifetime_params()
                    .map(|lifetime_param| {
                        lifetime_param
                            .lifetime()
                            .map_or_else(Name::missing, |lt| Name::new_lifetime(&lt))
                    })
                    .collect()
            });
            for bound in pred.type_bound_list().iter().flat_map(|l| l.bounds()) {
                self.add_where_predicate_from_bound(
                    lower_ctx,
                    bound,
                    lifetimes.as_ref(),
                    target.clone(),
                );
            }
        }
    }

    fn add_where_predicate_from_bound(
        &mut self,
        lower_ctx: &LowerCtx<'_>,
        bound: ast::TypeBound,
        hrtb_lifetimes: Option<&Box<[Name]>>,
        target: Either<TypeRef, LifetimeRef>,
    ) {
        let bound = TypeBound::from_ast(lower_ctx, bound);
        let predicate = match (target, bound) {
            (Either::Left(type_ref), bound) => match hrtb_lifetimes {
                Some(hrtb_lifetimes) => WherePredicate::ForLifetime {
                    lifetimes: hrtb_lifetimes.clone(),
                    target: WherePredicateTypeTarget::TypeRef(Interned::new(type_ref)),
                    bound: Interned::new(bound),
                },
                None => WherePredicate::TypeBound {
                    target: WherePredicateTypeTarget::TypeRef(Interned::new(type_ref)),
                    bound: Interned::new(bound),
                },
            },
            (Either::Right(lifetime), TypeBound::Lifetime(bound)) => {
                WherePredicate::Lifetime { target: lifetime, bound }
            }
            _ => return,
        };
        self.where_predicates.push(predicate);
    }

    pub(crate) fn fill_implicit_impl_trait_args(
        &mut self,
        db: &dyn DefDatabase,
        exp: &mut Lazy<(Arc<DefMap>, Expander), impl FnOnce() -> (Arc<DefMap>, Expander)>,
        type_ref: &TypeRef,
    ) {
        type_ref.walk(&mut |type_ref| {
            if let TypeRef::ImplTrait(bounds) = type_ref {
                let param = TypeParamData {
                    name: None,
                    default: None,
                    provenance: TypeParamProvenance::ArgumentImplTrait,
                };
                let param_id = self.type_or_consts.alloc(param.into());
                for bound in bounds {
                    self.where_predicates.push(WherePredicate::TypeBound {
                        target: WherePredicateTypeTarget::TypeOrConstParam(param_id),
                        bound: bound.clone(),
                    });
                }
            }
            if let TypeRef::Macro(mc) = type_ref {
                let macro_call = mc.to_node(db.upcast());
                let (def_map, expander) = &mut **exp;

                let module = expander.module.local_id;
                let resolver = |path| {
                    def_map
                        .resolve_path(
                            db,
                            module,
                            &path,
                            crate::item_scope::BuiltinShadowMode::Other,
                            Some(MacroSubNs::Bang),
                        )
                        .0
                        .take_macros()
                };
                if let Ok(ExpandResult { value: Some((mark, expanded)), .. }) =
                    expander.enter_expand(db, macro_call, resolver)
                {
                    let ctx = expander.ctx(db);
                    let type_ref = TypeRef::from_ast(&ctx, expanded.tree());
                    self.fill_implicit_impl_trait_args(db, &mut *exp, &type_ref);
                    exp.1.exit(db, mark);
                }
            }
        });
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        let Self { lifetimes, type_or_consts: types, where_predicates } = self;
        lifetimes.shrink_to_fit();
        types.shrink_to_fit();
        where_predicates.shrink_to_fit();
    }

    pub fn find_type_by_name(&self, name: &Name, parent: GenericDefId) -> Option<TypeParamId> {
        self.type_or_consts.iter().find_map(|(id, p)| {
            if p.name().as_ref() == Some(&name) && p.type_param().is_some() {
                Some(TypeParamId::from_unchecked(TypeOrConstParamId { local_id: id, parent }))
            } else {
                None
            }
        })
    }

    pub fn find_const_by_name(&self, name: &Name, parent: GenericDefId) -> Option<ConstParamId> {
        self.type_or_consts.iter().find_map(|(id, p)| {
            if p.name().as_ref() == Some(&name) && p.const_param().is_some() {
                Some(ConstParamId::from_unchecked(TypeOrConstParamId { local_id: id, parent }))
            } else {
                None
            }
        })
    }

    pub fn find_trait_self_param(&self) -> Option<LocalTypeOrConstParamId> {
        self.type_or_consts.iter().find_map(|(id, p)| {
            matches!(
                p,
                TypeOrConstParamData::TypeParamData(TypeParamData {
                    provenance: TypeParamProvenance::TraitSelf,
                    ..
                })
            )
            .then(|| id)
        })
    }
}

fn file_id_and_params_of(
    def: GenericDefId,
    db: &dyn DefDatabase,
) -> (HirFileId, Option<ast::GenericParamList>) {
    match def {
        GenericDefId::FunctionId(it) => {
            let src = it.lookup(db).source(db);
            (src.file_id, src.value.generic_param_list())
        }
        GenericDefId::AdtId(AdtId::StructId(it)) => {
            let src = it.lookup(db).source(db);
            (src.file_id, src.value.generic_param_list())
        }
        GenericDefId::AdtId(AdtId::UnionId(it)) => {
            let src = it.lookup(db).source(db);
            (src.file_id, src.value.generic_param_list())
        }
        GenericDefId::AdtId(AdtId::EnumId(it)) => {
            let src = it.lookup(db).source(db);
            (src.file_id, src.value.generic_param_list())
        }
        GenericDefId::TraitId(it) => {
            let src = it.lookup(db).source(db);
            (src.file_id, src.value.generic_param_list())
        }
        GenericDefId::TraitAliasId(it) => {
            let src = it.lookup(db).source(db);
            (src.file_id, src.value.generic_param_list())
        }
        GenericDefId::TypeAliasId(it) => {
            let src = it.lookup(db).source(db);
            (src.file_id, src.value.generic_param_list())
        }
        GenericDefId::ImplId(it) => {
            let src = it.lookup(db).source(db);
            (src.file_id, src.value.generic_param_list())
        }
        // We won't be using this ID anyway
        GenericDefId::EnumVariantId(_) | GenericDefId::ConstId(_) => (FileId(!0).into(), None),
    }
}

impl HasChildSource<LocalTypeOrConstParamId> for GenericDefId {
    type Value = Either<ast::TypeOrConstParam, ast::TraitOrAlias>;
    fn child_source(
        &self,
        db: &dyn DefDatabase,
    ) -> InFile<ArenaMap<LocalTypeOrConstParamId, Self::Value>> {
        let generic_params = db.generic_params(*self);
        let mut idx_iter = generic_params.type_or_consts.iter().map(|(idx, _)| idx);

        let (file_id, generic_params_list) = file_id_and_params_of(*self, db);

        let mut params = ArenaMap::default();

        // For traits and trait aliases the first type index is `Self`, we need to add it before
        // the other params.
        match *self {
            GenericDefId::TraitId(id) => {
                let trait_ref = id.lookup(db).source(db).value;
                let idx = idx_iter.next().unwrap();
                params.insert(idx, Either::Right(ast::TraitOrAlias::Trait(trait_ref)));
            }
            GenericDefId::TraitAliasId(id) => {
                let alias = id.lookup(db).source(db).value;
                let idx = idx_iter.next().unwrap();
                params.insert(idx, Either::Right(ast::TraitOrAlias::TraitAlias(alias)));
            }
            _ => {}
        }

        if let Some(generic_params_list) = generic_params_list {
            for (idx, ast_param) in idx_iter.zip(generic_params_list.type_or_const_params()) {
                params.insert(idx, Either::Left(ast_param));
            }
        }

        InFile::new(file_id, params)
    }
}

impl HasChildSource<LocalLifetimeParamId> for GenericDefId {
    type Value = ast::LifetimeParam;
    fn child_source(
        &self,
        db: &dyn DefDatabase,
    ) -> InFile<ArenaMap<LocalLifetimeParamId, Self::Value>> {
        let generic_params = db.generic_params(*self);
        let idx_iter = generic_params.lifetimes.iter().map(|(idx, _)| idx);

        let (file_id, generic_params_list) = file_id_and_params_of(*self, db);

        let mut params = ArenaMap::default();

        if let Some(generic_params_list) = generic_params_list {
            for (idx, ast_param) in idx_iter.zip(generic_params_list.lifetime_params()) {
                params.insert(idx, ast_param);
            }
        }

        InFile::new(file_id, params)
    }
}

impl ChildBySource for GenericDefId {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId) {
        let (gfile_id, generic_params_list) = file_id_and_params_of(*self, db);
        if gfile_id != file_id {
            return;
        }

        let generic_params = db.generic_params(*self);
        let mut toc_idx_iter = generic_params.type_or_consts.iter().map(|(idx, _)| idx);
        let lts_idx_iter = generic_params.lifetimes.iter().map(|(idx, _)| idx);

        // For traits the first type index is `Self`, skip it.
        if let GenericDefId::TraitId(_) = *self {
            toc_idx_iter.next().unwrap(); // advance_by(1);
        }

        if let Some(generic_params_list) = generic_params_list {
            for (local_id, ast_param) in
                toc_idx_iter.zip(generic_params_list.type_or_const_params())
            {
                let id = TypeOrConstParamId { parent: *self, local_id };
                match ast_param {
                    ast::TypeOrConstParam::Type(a) => res[keys::TYPE_PARAM].insert(a, id),
                    ast::TypeOrConstParam::Const(a) => res[keys::CONST_PARAM].insert(a, id),
                }
            }
            for (local_id, ast_param) in lts_idx_iter.zip(generic_params_list.lifetime_params()) {
                let id = LifetimeParamId { parent: *self, local_id };
                res[keys::LIFETIME_PARAM].insert(ast_param, id);
            }
        }
    }
}
