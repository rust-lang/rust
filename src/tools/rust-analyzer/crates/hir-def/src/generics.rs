//! Many kinds of items or constructs can have generic parameters: functions,
//! structs, impls, traits, etc. This module provides a common HIR for these
//! generic parameters. See also the `Generics` type and the `generics_of` query
//! in rustc.

use std::{ops, sync::LazyLock};

use either::Either;
use hir_expand::{
    name::{AsName, Name},
    ExpandResult,
};
use la_arena::{Arena, RawIdx};
use stdx::{
    impl_from,
    thin_vec::{EmptyOptimizedThinVec, ThinVec},
};
use syntax::ast::{self, HasGenericParams, HasName, HasTypeBounds};
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    expander::Expander,
    item_tree::{AttrOwner, FileItemTreeId, GenericModItem, GenericsItemTreeNode, ItemTree},
    lower::LowerCtx,
    nameres::{DefMap, MacroSubNs},
    path::{AssociatedTypeBinding, GenericArg, GenericArgs, NormalPath, Path},
    type_ref::{
        ArrayType, ConstRef, FnType, LifetimeRef, PathId, RefType, TypeBound, TypeRef, TypeRefId,
        TypesMap, TypesSourceMap,
    },
    AdtId, ConstParamId, GenericDefId, HasModule, ItemTreeLoc, LifetimeParamId,
    LocalLifetimeParamId, LocalTypeOrConstParamId, Lookup, TypeOrConstParamId, TypeParamId,
};

/// The index of the self param in the generic of the non-parent definition.
const SELF_PARAM_ID_IN_SELF: la_arena::Idx<TypeOrConstParamData> =
    LocalTypeOrConstParamId::from_raw(RawIdx::from_u32(0));

/// Data about a generic type parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct TypeParamData {
    /// [`None`] only if the type ref is an [`TypeRef::ImplTrait`]. FIXME: Might be better to just
    /// make it always be a value, giving impl trait a special name.
    pub name: Option<Name>,
    pub default: Option<TypeRefId>,
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
    pub ty: TypeRefId,
    pub default: Option<ConstRef>,
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
            TypeOrConstParamData::ConstParamData(it) => it.default.is_some(),
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

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum GenericParamData {
    TypeParamData(TypeParamData),
    ConstParamData(ConstParamData),
    LifetimeParamData(LifetimeParamData),
}

impl GenericParamData {
    pub fn name(&self) -> Option<&Name> {
        match self {
            GenericParamData::TypeParamData(it) => it.name.as_ref(),
            GenericParamData::ConstParamData(it) => Some(&it.name),
            GenericParamData::LifetimeParamData(it) => Some(&it.name),
        }
    }

    pub fn type_param(&self) -> Option<&TypeParamData> {
        match self {
            GenericParamData::TypeParamData(it) => Some(it),
            _ => None,
        }
    }

    pub fn const_param(&self) -> Option<&ConstParamData> {
        match self {
            GenericParamData::ConstParamData(it) => Some(it),
            _ => None,
        }
    }

    pub fn lifetime_param(&self) -> Option<&LifetimeParamData> {
        match self {
            GenericParamData::LifetimeParamData(it) => Some(it),
            _ => None,
        }
    }
}

impl_from!(TypeParamData, ConstParamData, LifetimeParamData for GenericParamData);

pub enum GenericParamDataRef<'a> {
    TypeParamData(&'a TypeParamData),
    ConstParamData(&'a ConstParamData),
    LifetimeParamData(&'a LifetimeParamData),
}

/// Data about the generic parameters of a function, struct, impl, etc.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct GenericParams {
    type_or_consts: Arena<TypeOrConstParamData>,
    lifetimes: Arena<LifetimeParamData>,
    where_predicates: Box<[WherePredicate]>,
    pub types_map: TypesMap,
}

impl ops::Index<LocalTypeOrConstParamId> for GenericParams {
    type Output = TypeOrConstParamData;
    fn index(&self, index: LocalTypeOrConstParamId) -> &TypeOrConstParamData {
        &self.type_or_consts[index]
    }
}

impl ops::Index<LocalLifetimeParamId> for GenericParams {
    type Output = LifetimeParamData;
    fn index(&self, index: LocalLifetimeParamId) -> &LifetimeParamData {
        &self.lifetimes[index]
    }
}

/// A single predicate from a where clause, i.e. `where Type: Trait`. Combined
/// where clauses like `where T: Foo + Bar` are turned into multiple of these.
/// It might still result in multiple actual predicates though, because of
/// associated type bindings like `Iterator<Item = u32>`.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum WherePredicate {
    TypeBound { target: WherePredicateTypeTarget, bound: TypeBound },
    Lifetime { target: LifetimeRef, bound: LifetimeRef },
    ForLifetime { lifetimes: Box<[Name]>, target: WherePredicateTypeTarget, bound: TypeBound },
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum WherePredicateTypeTarget {
    TypeRef(TypeRefId),
    /// For desugared where predicates that can directly refer to a type param.
    TypeOrConstParam(LocalTypeOrConstParamId),
}

impl GenericParams {
    /// Number of Generic parameters (type_or_consts + lifetimes)
    #[inline]
    pub fn len(&self) -> usize {
        self.type_or_consts.len() + self.lifetimes.len()
    }

    #[inline]
    pub fn len_lifetimes(&self) -> usize {
        self.lifetimes.len()
    }

    #[inline]
    pub fn len_type_or_consts(&self) -> usize {
        self.type_or_consts.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn no_predicates(&self) -> bool {
        self.where_predicates.is_empty()
    }

    #[inline]
    pub fn where_predicates(&self) -> std::slice::Iter<'_, WherePredicate> {
        self.where_predicates.iter()
    }

    /// Iterator of type_or_consts field
    #[inline]
    pub fn iter_type_or_consts(
        &self,
    ) -> impl DoubleEndedIterator<Item = (LocalTypeOrConstParamId, &TypeOrConstParamData)> {
        self.type_or_consts.iter()
    }

    /// Iterator of lifetimes field
    #[inline]
    pub fn iter_lt(
        &self,
    ) -> impl DoubleEndedIterator<Item = (LocalLifetimeParamId, &LifetimeParamData)> {
        self.lifetimes.iter()
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

    #[inline]
    pub fn trait_self_param(&self) -> Option<LocalTypeOrConstParamId> {
        if self.type_or_consts.is_empty() {
            return None;
        }
        matches!(
            self.type_or_consts[SELF_PARAM_ID_IN_SELF],
            TypeOrConstParamData::TypeParamData(TypeParamData {
                provenance: TypeParamProvenance::TraitSelf,
                ..
            })
        )
        .then(|| SELF_PARAM_ID_IN_SELF)
    }

    pub fn find_lifetime_by_name(
        &self,
        name: &Name,
        parent: GenericDefId,
    ) -> Option<LifetimeParamId> {
        self.lifetimes.iter().find_map(|(id, p)| {
            if &p.name == name {
                Some(LifetimeParamId { local_id: id, parent })
            } else {
                None
            }
        })
    }

    pub(crate) fn generic_params_query(
        db: &dyn DefDatabase,
        def: GenericDefId,
    ) -> Arc<GenericParams> {
        db.generic_params_with_source_map(def).0
    }

    pub(crate) fn generic_params_with_source_map_query(
        db: &dyn DefDatabase,
        def: GenericDefId,
    ) -> (Arc<GenericParams>, Option<Arc<TypesSourceMap>>) {
        let _p = tracing::info_span!("generic_params_query").entered();

        let krate = def.krate(db);
        let cfg_options = db.crate_graph();
        let cfg_options = &cfg_options[krate].cfg_options;

        // Returns the generic parameters that are enabled under the current `#[cfg]` options
        let enabled_params =
            |params: &Arc<GenericParams>, item_tree: &ItemTree, parent: GenericModItem| {
                let enabled = |param| item_tree.attrs(db, krate, param).is_cfg_enabled(cfg_options);
                let attr_owner_ct = |param| AttrOwner::TypeOrConstParamData(parent, param);
                let attr_owner_lt = |param| AttrOwner::LifetimeParamData(parent, param);

                // In the common case, no parameters will by disabled by `#[cfg]` attributes.
                // Therefore, make a first pass to check if all parameters are enabled and, if so,
                // clone the `Interned<GenericParams>` instead of recreating an identical copy.
                let all_type_or_consts_enabled =
                    params.type_or_consts.iter().all(|(idx, _)| enabled(attr_owner_ct(idx)));
                let all_lifetimes_enabled =
                    params.lifetimes.iter().all(|(idx, _)| enabled(attr_owner_lt(idx)));

                if all_type_or_consts_enabled && all_lifetimes_enabled {
                    params.clone()
                } else {
                    Arc::new(GenericParams {
                        type_or_consts: all_type_or_consts_enabled
                            .then(|| params.type_or_consts.clone())
                            .unwrap_or_else(|| {
                                params
                                    .type_or_consts
                                    .iter()
                                    .filter(|&(idx, _)| enabled(attr_owner_ct(idx)))
                                    .map(|(_, param)| param.clone())
                                    .collect()
                            }),
                        lifetimes: all_lifetimes_enabled
                            .then(|| params.lifetimes.clone())
                            .unwrap_or_else(|| {
                                params
                                    .lifetimes
                                    .iter()
                                    .filter(|&(idx, _)| enabled(attr_owner_lt(idx)))
                                    .map(|(_, param)| param.clone())
                                    .collect()
                            }),
                        where_predicates: params.where_predicates.clone(),
                        types_map: params.types_map.clone(),
                    })
                }
            };
        fn id_to_generics<Id: GenericsItemTreeNode>(
            db: &dyn DefDatabase,
            id: impl for<'db> Lookup<
                Database<'db> = dyn DefDatabase + 'db,
                Data = impl ItemTreeLoc<Id = Id>,
            >,
            enabled_params: impl Fn(
                &Arc<GenericParams>,
                &ItemTree,
                GenericModItem,
            ) -> Arc<GenericParams>,
        ) -> (Arc<GenericParams>, Option<Arc<TypesSourceMap>>)
        where
            FileItemTreeId<Id>: Into<GenericModItem>,
        {
            let id = id.lookup(db).item_tree_id();
            let tree = id.item_tree(db);
            let item = &tree[id.value];
            (enabled_params(item.generic_params(), &tree, id.value.into()), None)
        }

        match def {
            GenericDefId::FunctionId(id) => {
                let loc = id.lookup(db);
                let tree = loc.id.item_tree(db);
                let item = &tree[loc.id.value];

                let enabled_params =
                    enabled_params(&item.explicit_generic_params, &tree, loc.id.value.into());

                let module = loc.container.module(db);
                let func_data = db.function_data(id);
                if func_data.params.is_empty() {
                    (enabled_params, None)
                } else {
                    let source_maps = loc.id.item_tree_with_source_map(db).1;
                    let item_source_maps = source_maps.function(loc.id.value);
                    let mut generic_params = GenericParamsCollector {
                        type_or_consts: enabled_params.type_or_consts.clone(),
                        lifetimes: enabled_params.lifetimes.clone(),
                        where_predicates: enabled_params.where_predicates.clone().into(),
                    };

                    let (mut types_map, mut types_source_maps) =
                        (enabled_params.types_map.clone(), item_source_maps.generics().clone());
                    // Don't create an `Expander` if not needed since this
                    // could cause a reparse after the `ItemTree` has been created due to the spanmap.
                    let mut expander = None;
                    for &param in func_data.params.iter() {
                        generic_params.fill_implicit_impl_trait_args(
                            db,
                            &mut types_map,
                            &mut types_source_maps,
                            &mut expander,
                            &mut || {
                                (module.def_map(db), Expander::new(db, loc.id.file_id(), module))
                            },
                            param,
                            &item.types_map,
                            item_source_maps.item(),
                        );
                    }
                    let generics = generic_params.finish(types_map, &mut types_source_maps);
                    (generics, Some(Arc::new(types_source_maps)))
                }
            }
            GenericDefId::AdtId(AdtId::StructId(id)) => id_to_generics(db, id, enabled_params),
            GenericDefId::AdtId(AdtId::EnumId(id)) => id_to_generics(db, id, enabled_params),
            GenericDefId::AdtId(AdtId::UnionId(id)) => id_to_generics(db, id, enabled_params),
            GenericDefId::TraitId(id) => id_to_generics(db, id, enabled_params),
            GenericDefId::TraitAliasId(id) => id_to_generics(db, id, enabled_params),
            GenericDefId::TypeAliasId(id) => id_to_generics(db, id, enabled_params),
            GenericDefId::ImplId(id) => id_to_generics(db, id, enabled_params),
            GenericDefId::ConstId(_) | GenericDefId::StaticId(_) => (
                Arc::new(GenericParams {
                    type_or_consts: Default::default(),
                    lifetimes: Default::default(),
                    where_predicates: Default::default(),
                    types_map: Default::default(),
                }),
                None,
            ),
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct GenericParamsCollector {
    pub(crate) type_or_consts: Arena<TypeOrConstParamData>,
    lifetimes: Arena<LifetimeParamData>,
    where_predicates: Vec<WherePredicate>,
}

impl GenericParamsCollector {
    pub(crate) fn fill(
        &mut self,
        lower_ctx: &mut LowerCtx<'_>,
        node: &dyn HasGenericParams,
        add_param_attrs: impl FnMut(
            Either<LocalTypeOrConstParamId, LocalLifetimeParamId>,
            ast::GenericParam,
        ),
    ) {
        if let Some(params) = node.generic_param_list() {
            self.fill_params(lower_ctx, params, add_param_attrs)
        }
        if let Some(where_clause) = node.where_clause() {
            self.fill_where_predicates(lower_ctx, where_clause);
        }
    }

    pub(crate) fn fill_bounds(
        &mut self,
        lower_ctx: &mut LowerCtx<'_>,
        type_bounds: Option<ast::TypeBoundList>,
        target: Either<TypeRefId, LifetimeRef>,
    ) {
        for bound in type_bounds.iter().flat_map(|type_bound_list| type_bound_list.bounds()) {
            self.add_where_predicate_from_bound(lower_ctx, bound, None, target.clone());
        }
    }

    fn fill_params(
        &mut self,
        lower_ctx: &mut LowerCtx<'_>,
        params: ast::GenericParamList,
        mut add_param_attrs: impl FnMut(
            Either<LocalTypeOrConstParamId, LocalLifetimeParamId>,
            ast::GenericParam,
        ),
    ) {
        for type_or_const_param in params.type_or_const_params() {
            match type_or_const_param {
                ast::TypeOrConstParam::Type(type_param) => {
                    let name = type_param.name().map_or_else(Name::missing, |it| it.as_name());
                    // FIXME: Use `Path::from_src`
                    let default =
                        type_param.default_type().map(|it| TypeRef::from_ast(lower_ctx, it));
                    let param = TypeParamData {
                        name: Some(name.clone()),
                        default,
                        provenance: TypeParamProvenance::TypeParamList,
                    };
                    let idx = self.type_or_consts.alloc(param.into());
                    let type_ref = lower_ctx.alloc_type_ref_desugared(TypeRef::Path(name.into()));
                    self.fill_bounds(
                        lower_ctx,
                        type_param.type_bound_list(),
                        Either::Left(type_ref),
                    );
                    add_param_attrs(Either::Left(idx), ast::GenericParam::TypeParam(type_param));
                }
                ast::TypeOrConstParam::Const(const_param) => {
                    let name = const_param.name().map_or_else(Name::missing, |it| it.as_name());
                    let ty = TypeRef::from_ast_opt(lower_ctx, const_param.ty());
                    let param = ConstParamData {
                        name,
                        ty,
                        default: ConstRef::from_const_param(lower_ctx, &const_param),
                    };
                    let idx = self.type_or_consts.alloc(param.into());
                    add_param_attrs(Either::Left(idx), ast::GenericParam::ConstParam(const_param));
                }
            }
        }
        for lifetime_param in params.lifetime_params() {
            let name =
                lifetime_param.lifetime().map_or_else(Name::missing, |lt| Name::new_lifetime(&lt));
            let param = LifetimeParamData { name: name.clone() };
            let idx = self.lifetimes.alloc(param);
            let lifetime_ref = LifetimeRef::new_name(name);
            self.fill_bounds(
                lower_ctx,
                lifetime_param.type_bound_list(),
                Either::Right(lifetime_ref),
            );
            add_param_attrs(Either::Right(idx), ast::GenericParam::LifetimeParam(lifetime_param));
        }
    }

    fn fill_where_predicates(
        &mut self,
        lower_ctx: &mut LowerCtx<'_>,
        where_clause: ast::WhereClause,
    ) {
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
                    lifetimes.as_deref(),
                    target.clone(),
                );
            }
        }
    }

    fn add_where_predicate_from_bound(
        &mut self,
        lower_ctx: &mut LowerCtx<'_>,
        bound: ast::TypeBound,
        hrtb_lifetimes: Option<&[Name]>,
        target: Either<TypeRefId, LifetimeRef>,
    ) {
        let bound = TypeBound::from_ast(lower_ctx, bound);
        self.fill_impl_trait_bounds(lower_ctx.take_impl_traits_bounds());
        let predicate = match (target, bound) {
            (Either::Left(type_ref), bound) => match hrtb_lifetimes {
                Some(hrtb_lifetimes) => WherePredicate::ForLifetime {
                    lifetimes: hrtb_lifetimes.to_vec().into_boxed_slice(),
                    target: WherePredicateTypeTarget::TypeRef(type_ref),
                    bound,
                },
                None => WherePredicate::TypeBound {
                    target: WherePredicateTypeTarget::TypeRef(type_ref),
                    bound,
                },
            },
            (Either::Right(lifetime), TypeBound::Lifetime(bound)) => {
                WherePredicate::Lifetime { target: lifetime, bound }
            }
            _ => return,
        };
        self.where_predicates.push(predicate);
    }

    fn fill_impl_trait_bounds(&mut self, impl_bounds: Vec<ThinVec<TypeBound>>) {
        for bounds in impl_bounds {
            let param = TypeParamData {
                name: None,
                default: None,
                provenance: TypeParamProvenance::ArgumentImplTrait,
            };
            let param_id = self.type_or_consts.alloc(param.into());
            for bound in &bounds {
                self.where_predicates.push(WherePredicate::TypeBound {
                    target: WherePredicateTypeTarget::TypeOrConstParam(param_id),
                    bound: bound.clone(),
                });
            }
        }
    }

    fn fill_implicit_impl_trait_args(
        &mut self,
        db: &dyn DefDatabase,
        generics_types_map: &mut TypesMap,
        generics_types_source_map: &mut TypesSourceMap,
        // FIXME: Change this back to `LazyCell` if https://github.com/rust-lang/libs-team/issues/429 is accepted.
        exp: &mut Option<(Arc<DefMap>, Expander)>,
        exp_fill: &mut dyn FnMut() -> (Arc<DefMap>, Expander),
        type_ref: TypeRefId,
        types_map: &TypesMap,
        types_source_map: &TypesSourceMap,
    ) {
        TypeRef::walk(type_ref, types_map, &mut |type_ref| {
            if let TypeRef::ImplTrait(bounds) = type_ref {
                let param = TypeParamData {
                    name: None,
                    default: None,
                    provenance: TypeParamProvenance::ArgumentImplTrait,
                };
                let param_id = self.type_or_consts.alloc(param.into());
                for bound in bounds {
                    let bound = copy_type_bound(
                        bound,
                        types_map,
                        types_source_map,
                        generics_types_map,
                        generics_types_source_map,
                    );
                    self.where_predicates.push(WherePredicate::TypeBound {
                        target: WherePredicateTypeTarget::TypeOrConstParam(param_id),
                        bound,
                    });
                }
            }

            if let TypeRef::Macro(mc) = type_ref {
                let macro_call = mc.to_node(db.upcast());
                let (def_map, expander) = exp.get_or_insert_with(&mut *exp_fill);

                let module = expander.module.local_id;
                let resolver = |path: &_| {
                    def_map
                        .resolve_path(
                            db,
                            module,
                            path,
                            crate::item_scope::BuiltinShadowMode::Other,
                            Some(MacroSubNs::Bang),
                        )
                        .0
                        .take_macros()
                };
                if let Ok(ExpandResult { value: Some((mark, expanded)), .. }) =
                    expander.enter_expand(db, macro_call, resolver)
                {
                    let (mut macro_types_map, mut macro_types_source_map) =
                        (TypesMap::default(), TypesSourceMap::default());
                    let mut ctx =
                        expander.ctx(db, &mut macro_types_map, &mut macro_types_source_map);
                    let type_ref = TypeRef::from_ast(&mut ctx, expanded.tree());
                    self.fill_implicit_impl_trait_args(
                        db,
                        generics_types_map,
                        generics_types_source_map,
                        &mut *exp,
                        exp_fill,
                        type_ref,
                        &macro_types_map,
                        &macro_types_source_map,
                    );
                    exp.get_or_insert_with(&mut *exp_fill).1.exit(mark);
                }
            }
        });
    }

    pub(crate) fn finish(
        self,
        mut generics_types_map: TypesMap,
        generics_types_source_map: &mut TypesSourceMap,
    ) -> Arc<GenericParams> {
        let Self { mut lifetimes, mut type_or_consts, mut where_predicates } = self;

        if lifetimes.is_empty() && type_or_consts.is_empty() && where_predicates.is_empty() {
            static EMPTY: LazyLock<Arc<GenericParams>> = LazyLock::new(|| {
                Arc::new(GenericParams {
                    lifetimes: Arena::new(),
                    type_or_consts: Arena::new(),
                    where_predicates: Box::default(),
                    types_map: TypesMap::default(),
                })
            });
            return Arc::clone(&EMPTY);
        }

        lifetimes.shrink_to_fit();
        type_or_consts.shrink_to_fit();
        where_predicates.shrink_to_fit();
        generics_types_map.shrink_to_fit();
        generics_types_source_map.shrink_to_fit();
        Arc::new(GenericParams {
            type_or_consts,
            lifetimes,
            where_predicates: where_predicates.into_boxed_slice(),
            types_map: generics_types_map,
        })
    }
}

/// Copies a `TypeRef` from a `TypesMap` (accompanied with `TypesSourceMap`) into another `TypesMap`
/// (and `TypesSourceMap`).
fn copy_type_ref(
    type_ref: TypeRefId,
    from: &TypesMap,
    from_source_map: &TypesSourceMap,
    to: &mut TypesMap,
    to_source_map: &mut TypesSourceMap,
) -> TypeRefId {
    let result = match &from[type_ref] {
        TypeRef::Fn(fn_) => {
            let params = fn_.params().iter().map(|(name, param_type)| {
                (name.clone(), copy_type_ref(*param_type, from, from_source_map, to, to_source_map))
            });
            TypeRef::Fn(FnType::new(fn_.is_varargs(), fn_.is_unsafe(), fn_.abi().clone(), params))
        }
        TypeRef::Tuple(types) => TypeRef::Tuple(EmptyOptimizedThinVec::from_iter(
            types.iter().map(|&t| copy_type_ref(t, from, from_source_map, to, to_source_map)),
        )),
        &TypeRef::RawPtr(type_ref, mutbl) => TypeRef::RawPtr(
            copy_type_ref(type_ref, from, from_source_map, to, to_source_map),
            mutbl,
        ),
        TypeRef::Reference(ref_) => TypeRef::Reference(Box::new(RefType {
            ty: copy_type_ref(ref_.ty, from, from_source_map, to, to_source_map),
            lifetime: ref_.lifetime.clone(),
            mutability: ref_.mutability,
        })),
        TypeRef::Array(array) => TypeRef::Array(Box::new(ArrayType {
            ty: copy_type_ref(array.ty, from, from_source_map, to, to_source_map),
            len: array.len.clone(),
        })),
        &TypeRef::Slice(type_ref) => {
            TypeRef::Slice(copy_type_ref(type_ref, from, from_source_map, to, to_source_map))
        }
        TypeRef::ImplTrait(bounds) => TypeRef::ImplTrait(ThinVec::from_iter(copy_type_bounds(
            bounds,
            from,
            from_source_map,
            to,
            to_source_map,
        ))),
        TypeRef::DynTrait(bounds) => TypeRef::DynTrait(ThinVec::from_iter(copy_type_bounds(
            bounds,
            from,
            from_source_map,
            to,
            to_source_map,
        ))),
        TypeRef::Path(path) => {
            TypeRef::Path(copy_path(path, from, from_source_map, to, to_source_map))
        }
        TypeRef::Never => TypeRef::Never,
        TypeRef::Placeholder => TypeRef::Placeholder,
        TypeRef::Macro(macro_call) => TypeRef::Macro(*macro_call),
        TypeRef::Error => TypeRef::Error,
    };
    let id = to.types.alloc(result);
    if let Some(&ptr) = from_source_map.types_map_back.get(id) {
        to_source_map.types_map_back.insert(id, ptr);
    }
    id
}

fn copy_path(
    path: &Path,
    from: &TypesMap,
    from_source_map: &TypesSourceMap,
    to: &mut TypesMap,
    to_source_map: &mut TypesSourceMap,
) -> Path {
    match path {
        Path::BarePath(mod_path) => Path::BarePath(mod_path.clone()),
        Path::Normal(path) => {
            let type_anchor = path
                .type_anchor()
                .map(|type_ref| copy_type_ref(type_ref, from, from_source_map, to, to_source_map));
            let mod_path = path.mod_path().clone();
            let generic_args = path.generic_args().iter().map(|generic_args| {
                copy_generic_args(generic_args, from, from_source_map, to, to_source_map)
            });
            Path::Normal(NormalPath::new(type_anchor, mod_path, generic_args))
        }
        Path::LangItem(lang_item, name) => Path::LangItem(*lang_item, name.clone()),
    }
}

fn copy_generic_args(
    generic_args: &Option<GenericArgs>,
    from: &TypesMap,
    from_source_map: &TypesSourceMap,
    to: &mut TypesMap,
    to_source_map: &mut TypesSourceMap,
) -> Option<GenericArgs> {
    generic_args.as_ref().map(|generic_args| {
        let args = generic_args
            .args
            .iter()
            .map(|arg| match arg {
                &GenericArg::Type(ty) => {
                    GenericArg::Type(copy_type_ref(ty, from, from_source_map, to, to_source_map))
                }
                GenericArg::Lifetime(lifetime) => GenericArg::Lifetime(lifetime.clone()),
                GenericArg::Const(konst) => GenericArg::Const(konst.clone()),
            })
            .collect();
        let bindings = generic_args
            .bindings
            .iter()
            .map(|binding| {
                let name = binding.name.clone();
                let args =
                    copy_generic_args(&binding.args, from, from_source_map, to, to_source_map);
                let type_ref = binding.type_ref.map(|type_ref| {
                    copy_type_ref(type_ref, from, from_source_map, to, to_source_map)
                });
                let bounds =
                    copy_type_bounds(&binding.bounds, from, from_source_map, to, to_source_map)
                        .collect();
                AssociatedTypeBinding { name, args, type_ref, bounds }
            })
            .collect();
        GenericArgs {
            args,
            has_self_type: generic_args.has_self_type,
            bindings,
            desugared_from_fn: generic_args.desugared_from_fn,
        }
    })
}

fn copy_type_bounds<'a>(
    bounds: &'a [TypeBound],
    from: &'a TypesMap,
    from_source_map: &'a TypesSourceMap,
    to: &'a mut TypesMap,
    to_source_map: &'a mut TypesSourceMap,
) -> impl stdx::thin_vec::TrustedLen<Item = TypeBound> + 'a {
    bounds.iter().map(|bound| copy_type_bound(bound, from, from_source_map, to, to_source_map))
}

fn copy_type_bound(
    bound: &TypeBound,
    from: &TypesMap,
    from_source_map: &TypesSourceMap,
    to: &mut TypesMap,
    to_source_map: &mut TypesSourceMap,
) -> TypeBound {
    let mut copy_path_id = |path: PathId| {
        let new_path = copy_path(&from[path], from, from_source_map, to, to_source_map);
        let new_path_id = to.types.alloc(TypeRef::Path(new_path));
        if let Some(&ptr) = from_source_map.types_map_back.get(path.type_ref()) {
            to_source_map.types_map_back.insert(new_path_id, ptr);
        }
        PathId::from_type_ref_unchecked(new_path_id)
    };

    match bound {
        &TypeBound::Path(path, modifier) => TypeBound::Path(copy_path_id(path), modifier),
        TypeBound::ForLifetime(lifetimes, path) => {
            TypeBound::ForLifetime(lifetimes.clone(), copy_path_id(*path))
        }
        TypeBound::Lifetime(lifetime) => TypeBound::Lifetime(lifetime.clone()),
        TypeBound::Use(use_args) => TypeBound::Use(use_args.clone()),
        TypeBound::Error => TypeBound::Error,
    }
}
