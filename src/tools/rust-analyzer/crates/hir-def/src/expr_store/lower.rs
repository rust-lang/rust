//! Transforms `ast::Expr` into an equivalent `hir_def::expr::Expr`
//! representation.

mod asm;
mod format_args;
mod generics;
mod path;

use std::{cell::OnceCell, mem};

use arrayvec::ArrayVec;
use base_db::FxIndexSet;
use cfg::CfgOptions;
use either::Either;
use hir_expand::{
    HirFileId, InFile, MacroDefId,
    mod_path::ModPath,
    name::{AsName, Name},
    span_map::SpanMap,
};
use intern::{Symbol, sym};
use rustc_abi::ExternAbi;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use stdx::never;
use syntax::{
    AstNode, AstPtr, SyntaxNodePtr,
    ast::{
        self, ArrayExprKind, AstChildren, BlockExpr, HasArgList, HasAttrs, HasGenericArgs,
        HasGenericParams, HasLoopBody, HasName, HasTypeBounds, IsString, RangeItem,
        SlicePatComponents,
    },
};
use thin_vec::ThinVec;
use tt::TextRange;

use crate::{
    AdtId, BlockId, BlockLoc, ConstId, DefWithBodyId, FunctionId, GenericDefId, ImplId,
    ItemContainerId, MacroId, ModuleDefId, ModuleId, TraitId, TypeAliasId, UnresolvedMacro,
    attrs::AttrFlags,
    db::DefDatabase,
    expr_store::{
        Body, BodySourceMap, ExprPtr, ExprRoot, ExpressionStore, ExpressionStoreBuilder,
        ExpressionStoreDiagnostics, ExpressionStoreSourceMap, HygieneId, LabelPtr, LifetimePtr,
        PatPtr, TypePtr,
        expander::Expander,
        lower::generics::ImplTraitLowerFn,
        path::{AssociatedTypeBinding, GenericArg, GenericArgs, GenericArgsParentheses, Path},
    },
    hir::{
        Array, Binding, BindingAnnotation, BindingId, BindingProblems, CaptureBy, ClosureKind,
        CoroutineKind, CoroutineSource, Expr, ExprId, Item, Label, LabelId, Literal, MatchArm,
        Movability, OffsetOf, Pat, PatId, RecordFieldPat, RecordLitField, RecordSpread, Statement,
        generics::GenericParams,
    },
    item_scope::BuiltinShadowMode,
    item_tree::FieldsShape,
    lang_item::{LangItemTarget, LangItems},
    nameres::{DefMap, LocalDefMap, MacroSubNs, block_def_map},
    signatures::StructSignature,
    type_ref::{
        ArrayType, ConstRef, FnType, LifetimeRef, LifetimeRefId, Mutability, PathId, Rawness,
        RefType, TraitBoundModifier, TraitRef, TypeBound, TypeRef, TypeRefId, UseArgRef,
    },
};

pub use self::path::hir_segment_to_ast_segment;

pub(super) fn lower_body(
    db: &dyn DefDatabase,
    owner: DefWithBodyId,
    current_file_id: HirFileId,
    module: ModuleId,
    parameters: Option<ast::ParamList>,
    body: Option<ast::Expr>,
    is_async_fn: bool,
    is_gen_fn: bool,
) -> (Body, BodySourceMap) {
    // We cannot leave the root span map empty and let any identifier from it be treated as root,
    // because when inside nested macros `SyntaxContextId`s from the outer macro will be interleaved
    // with the inner macro, and that will cause confusion because they won't be the same as `ROOT`
    // even though they should be the same. Also, when the body comes from multiple expansions, their
    // hygiene is different.

    let mut self_params = ArrayVec::new();
    let mut source_map_self_param = None;
    let mut params = vec![];
    let mut collector = ExprCollector::new(db, module, current_file_id);

    let skip_body = AttrFlags::query(
        db,
        match owner {
            DefWithBodyId::FunctionId(it) => it.into(),
            DefWithBodyId::StaticId(it) => it.into(),
            DefWithBodyId::ConstId(it) => it.into(),
            DefWithBodyId::VariantId(it) => it.into(),
        },
    )
    .contains(AttrFlags::RUST_ANALYZER_SKIP);
    // If #[rust_analyzer::skip] annotated, only construct enough information for the signature
    // and skip the body.
    if skip_body {
        if let Some(param_list) = parameters {
            if let Some(self_param_syn) =
                param_list.self_param().filter(|self_param| collector.check_cfg(self_param))
            {
                let is_mutable =
                    self_param_syn.mut_token().is_some() && self_param_syn.amp_token().is_none();
                let hygiene = self_param_syn
                    .name()
                    .map(|name| collector.hygiene_id_for(name.syntax().text_range()))
                    .unwrap_or(HygieneId::ROOT);
                let binding_id: la_arena::Idx<Binding> = collector.alloc_binding(
                    Name::new_symbol_root(sym::self_),
                    BindingAnnotation::new(is_mutable, false),
                    hygiene,
                );
                self_params.push(binding_id);
                source_map_self_param =
                    Some(collector.expander.in_file(AstPtr::new(&self_param_syn)));
            }
            let count = param_list.params().filter(|it| collector.check_cfg(it)).count();
            params = (0..count).map(|_| collector.missing_pat()).collect();
        };
        collector.with_expr_root(|collector| collector.missing_expr());
        let (store, source_map) = collector.store.finish();
        return (
            Body { store, params: params.into_boxed_slice(), self_params },
            BodySourceMap { self_param: source_map_self_param, store: source_map },
        );
    }

    if let Some(param_list) = parameters {
        if let Some(self_param_syn) = param_list.self_param().filter(|it| collector.check_cfg(it)) {
            let is_mutable =
                self_param_syn.mut_token().is_some() && self_param_syn.amp_token().is_none();
            let hygiene = self_param_syn
                .name()
                .map(|name| collector.hygiene_id_for(name.syntax().text_range()))
                .unwrap_or(HygieneId::ROOT);
            let binding_id: la_arena::Idx<Binding> = collector.alloc_binding(
                Name::new_symbol_root(sym::self_),
                BindingAnnotation::new(is_mutable, false),
                hygiene,
            );
            self_params.push(binding_id);
            source_map_self_param = Some(collector.expander.in_file(AstPtr::new(&self_param_syn)));
        }

        let is_extern = matches!(
            owner,
            DefWithBodyId::FunctionId(id)
                if matches!(id.loc(db).container, ItemContainerId::ExternBlockId(_)),
        );

        for param in param_list.params() {
            if collector.check_cfg(&param) {
                let param_pat = if is_extern {
                    collector.collect_extern_fn_param(param.pat())
                } else {
                    collector.collect_pat_top(param.pat())
                };
                params.push(param_pat);
            }
        }
    };

    collector.with_expr_root(|collector| {
        collector.collect(
            &mut self_params,
            &mut params,
            body,
            if is_async_fn {
                Awaitable::Yes
            } else {
                match owner {
                    DefWithBodyId::FunctionId(..) => Awaitable::No("non-async function"),
                    DefWithBodyId::StaticId(..) => Awaitable::No("static"),
                    DefWithBodyId::ConstId(..) => Awaitable::No("constant"),
                    DefWithBodyId::VariantId(..) => Awaitable::No("enum variant"),
                }
            },
            is_async_fn,
            is_gen_fn,
        )
    });

    let (store, source_map) = collector.store.finish();
    (
        Body { store, params: params.into_boxed_slice(), self_params },
        BodySourceMap { self_param: source_map_self_param, store: source_map },
    )
}

pub(crate) fn lower_type_ref(
    db: &dyn DefDatabase,
    module: ModuleId,
    type_ref: InFile<Option<ast::Type>>,
) -> (ExpressionStore, ExpressionStoreSourceMap, TypeRefId) {
    let mut expr_collector = ExprCollector::new(db, module, type_ref.file_id);
    let type_ref =
        expr_collector.lower_type_ref_opt(type_ref.value, &mut ExprCollector::impl_trait_allocator);
    let (store, source_map) = expr_collector.store.finish();
    (store, source_map, type_ref)
}

pub(crate) fn lower_generic_params(
    db: &dyn DefDatabase,
    module: ModuleId,
    def: GenericDefId,
    file_id: HirFileId,
    param_list: Option<ast::GenericParamList>,
    where_clause: Option<ast::WhereClause>,
) -> (ExpressionStore, GenericParams, ExpressionStoreSourceMap) {
    let mut expr_collector = ExprCollector::new(db, module, file_id);
    let mut collector = generics::GenericParamsCollector::new(def);
    collector.lower(&mut expr_collector, param_list, where_clause);
    let params = collector.finish();
    let (store, source_map) = expr_collector.store.finish();
    (store, params, source_map)
}

pub(crate) fn lower_impl(
    db: &dyn DefDatabase,
    module: ModuleId,
    impl_syntax: InFile<ast::Impl>,
    impl_id: ImplId,
) -> (ExpressionStore, ExpressionStoreSourceMap, TypeRefId, Option<TraitRef>, GenericParams) {
    let mut expr_collector = ExprCollector::new(db, module, impl_syntax.file_id);
    let self_ty =
        expr_collector.lower_type_ref_opt_disallow_impl_trait(impl_syntax.value.self_ty());
    let trait_ = impl_syntax.value.trait_().and_then(|it| match &it {
        ast::Type::PathType(path_type) => {
            let path = expr_collector
                .lower_path_type(path_type, &mut ExprCollector::impl_trait_allocator)?;
            Some(TraitRef { path: expr_collector.alloc_path(path, AstPtr::new(&it)) })
        }
        _ => None,
    });
    let mut collector = generics::GenericParamsCollector::new(impl_id.into());
    collector.lower(
        &mut expr_collector,
        impl_syntax.value.generic_param_list(),
        impl_syntax.value.where_clause(),
    );
    let params = collector.finish();
    let (store, source_map) = expr_collector.store.finish();
    (store, source_map, self_ty, trait_, params)
}

pub(crate) fn lower_trait(
    db: &dyn DefDatabase,
    module: ModuleId,
    trait_syntax: InFile<ast::Trait>,
    trait_id: TraitId,
) -> (ExpressionStore, ExpressionStoreSourceMap, GenericParams) {
    let mut expr_collector = ExprCollector::new(db, module, trait_syntax.file_id);
    let mut collector = generics::GenericParamsCollector::with_self_param(
        &mut expr_collector,
        trait_id.into(),
        trait_syntax.value.type_bound_list(),
    );
    collector.lower(
        &mut expr_collector,
        trait_syntax.value.generic_param_list(),
        trait_syntax.value.where_clause(),
    );
    let params = collector.finish();
    let (store, source_map) = expr_collector.store.finish();
    (store, source_map, params)
}

pub(crate) fn lower_type_alias(
    db: &dyn DefDatabase,
    module: ModuleId,
    alias: InFile<ast::TypeAlias>,
    type_alias_id: TypeAliasId,
) -> (ExpressionStore, ExpressionStoreSourceMap, GenericParams, Box<[TypeBound]>, Option<TypeRefId>)
{
    let mut expr_collector = ExprCollector::new(db, module, alias.file_id);
    let bounds = alias
        .value
        .type_bound_list()
        .map(|bounds| {
            bounds
                .bounds()
                .map(|bound| {
                    expr_collector.lower_type_bound(bound, &mut ExprCollector::impl_trait_allocator)
                })
                .collect()
        })
        .unwrap_or_default();
    let mut collector = generics::GenericParamsCollector::new(type_alias_id.into());
    collector.lower(
        &mut expr_collector,
        alias.value.generic_param_list(),
        alias.value.where_clause(),
    );
    let params = collector.finish();
    let type_ref = alias
        .value
        .ty()
        .map(|ty| expr_collector.lower_type_ref(ty, &mut ExprCollector::impl_trait_allocator));
    let (store, source_map) = expr_collector.store.finish();
    (store, source_map, params, bounds, type_ref)
}

pub(crate) fn lower_function(
    db: &dyn DefDatabase,
    module: ModuleId,
    fn_: InFile<ast::Fn>,
    function_id: FunctionId,
) -> (
    ExpressionStore,
    ExpressionStoreSourceMap,
    GenericParams,
    Box<[TypeRefId]>,
    Option<TypeRefId>,
    bool,
    bool,
) {
    let mut expr_collector = ExprCollector::new(db, module, fn_.file_id);
    let mut collector = generics::GenericParamsCollector::new(function_id.into());
    collector.lower(&mut expr_collector, fn_.value.generic_param_list(), fn_.value.where_clause());
    let mut params = vec![];
    let mut has_self_param = false;
    let mut has_variadic = false;
    collector.collect_impl_trait(&mut expr_collector, |collector, mut impl_trait_lower_fn| {
        if let Some(param_list) = fn_.value.param_list() {
            if let Some(param) = param_list.self_param() {
                let enabled = collector.check_cfg(&param);
                if enabled {
                    has_self_param = true;
                    params.push(match param.ty() {
                        Some(ty) => collector.lower_type_ref(ty, &mut impl_trait_lower_fn),
                        None => {
                            let self_type = collector.alloc_type_ref_desugared(TypeRef::Path(
                                Name::new_symbol_root(sym::Self_).into(),
                            ));
                            let lifetime = param
                                .lifetime()
                                .map(|lifetime| collector.lower_lifetime_ref(lifetime));
                            match param.kind() {
                                ast::SelfParamKind::Owned => self_type,
                                ast::SelfParamKind::Ref => collector.alloc_type_ref_desugared(
                                    TypeRef::Reference(Box::new(RefType {
                                        ty: self_type,
                                        lifetime,
                                        mutability: Mutability::Shared,
                                    })),
                                ),
                                ast::SelfParamKind::MutRef => collector.alloc_type_ref_desugared(
                                    TypeRef::Reference(Box::new(RefType {
                                        ty: self_type,
                                        lifetime,
                                        mutability: Mutability::Mut,
                                    })),
                                ),
                            }
                        }
                    });
                }
            }
            let p = param_list
                .params()
                .filter(|param| collector.check_cfg(param))
                .filter(|param| {
                    let is_variadic = param.dotdotdot_token().is_some();
                    has_variadic |= is_variadic;
                    !is_variadic
                })
                .map(|param| param.ty())
                // FIXME
                .collect::<Vec<_>>();
            for p in p {
                params.push(collector.lower_type_ref_opt(p, &mut impl_trait_lower_fn));
            }
        }
    });
    let generics = collector.finish();
    let return_type = fn_.value.ret_type().map(|ret_type| {
        expr_collector.lower_type_ref_opt(ret_type.ty(), &mut ExprCollector::impl_trait_allocator)
    });

    let return_type = if fn_.value.async_token().is_some() || fn_.value.gen_token().is_some() {
        let (path, assoc_name) =
            match (fn_.value.async_token().is_some(), fn_.value.gen_token().is_some()) {
                (true, true) => {
                    (hir_expand::mod_path::path![core::async_iter::AsyncIterator], sym::Item)
                }
                (true, false) => (hir_expand::mod_path::path![core::future::Future], sym::Output),
                (false, true) => (hir_expand::mod_path::path![core::iter::Iterator], sym::Item),
                (false, false) => unreachable!(),
            };
        let mut generic_args: Vec<_> =
            std::iter::repeat_n(None, path.segments().len() - 1).collect();
        let binding = AssociatedTypeBinding {
            name: Name::new_symbol_root(assoc_name),
            args: None,
            type_ref: Some(
                return_type
                    .unwrap_or_else(|| expr_collector.alloc_type_ref_desugared(TypeRef::unit())),
            ),
            bounds: Box::default(),
        };
        generic_args
            .push(Some(GenericArgs { bindings: Box::new([binding]), ..GenericArgs::empty() }));

        let path = Path::from_known_path(path, generic_args);
        let path = PathId::from_type_ref_unchecked(
            expr_collector.alloc_type_ref_desugared(TypeRef::Path(path)),
        );
        let ty_bound = TypeBound::Path(path, TraitBoundModifier::None);
        Some(
            expr_collector
                .alloc_type_ref_desugared(TypeRef::ImplTrait(ThinVec::from_iter([ty_bound]))),
        )
    } else {
        return_type
    };
    let (store, source_map) = expr_collector.store.finish();
    (
        store,
        source_map,
        generics,
        params.into_boxed_slice(),
        return_type,
        has_self_param,
        has_variadic,
    )
}

pub struct ExprCollector<'db> {
    db: &'db dyn DefDatabase,
    cfg_options: &'db CfgOptions,
    expander: Expander<'db>,
    def_map: &'db DefMap,
    local_def_map: &'db LocalDefMap,
    module: ModuleId,
    lang_items: OnceCell<&'db LangItems>,
    pub store: ExpressionStoreBuilder,

    // state stuff
    // Prevent nested impl traits like `impl Foo<impl Bar>`.
    outer_impl_trait: bool,

    is_lowering_coroutine: bool,

    /// Legacy (`macro_rules!`) macros can have multiple definitions and shadow each other,
    /// and we need to find the current definition. So we track the number of definitions we saw.
    current_block_legacy_macro_defs_count: FxHashMap<Name, usize>,

    current_try_block: Option<TryBlock>,

    label_ribs: Vec<LabelRib>,
    unowned_bindings: Vec<BindingId>,

    awaitable_context: Option<Awaitable>,
    krate: base_db::Crate,

    name_generator_index: usize,
}

#[derive(Clone, Debug)]
struct LabelRib {
    kind: RibKind,
}

impl LabelRib {
    fn new(kind: RibKind) -> Self {
        LabelRib { kind }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RibKind {
    Normal(Name, LabelId, HygieneId),
    Closure,
    Constant,
    MacroDef(Box<MacroDefId>),
}

impl RibKind {
    /// This rib forbids referring to labels defined in upwards ribs.
    fn is_label_barrier(&self) -> bool {
        match self {
            RibKind::Normal(..) | RibKind::MacroDef(_) => false,
            RibKind::Closure | RibKind::Constant => true,
        }
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
enum Awaitable {
    Yes,
    No(&'static str),
}

enum TryBlock {
    // `try { ... }`
    Homogeneous { label: LabelId },
    // `try bikeshed Ty { ... }`
    Heterogeneous { label: LabelId },
}

#[derive(Debug, Default)]
struct BindingList {
    map: FxHashMap<(Name, HygieneId), BindingId>,
    is_used: FxHashMap<BindingId, bool>,
    reject_new: bool,
}

impl BindingList {
    fn find(
        &mut self,
        ec: &mut ExprCollector<'_>,
        name: Name,
        hygiene: HygieneId,
        mode: BindingAnnotation,
    ) -> BindingId {
        let id = *self
            .map
            .entry((name, hygiene))
            .or_insert_with_key(|(name, hygiene)| ec.alloc_binding(name.clone(), mode, *hygiene));
        if ec.store.bindings[id].mode != mode {
            ec.store.bindings[id].problems = Some(BindingProblems::BoundInconsistently);
        }
        self.check_is_used(ec, id);
        id
    }

    fn check_is_used(&mut self, ec: &mut ExprCollector<'_>, id: BindingId) {
        match self.is_used.get(&id) {
            None => {
                if self.reject_new {
                    ec.store.bindings[id].problems = Some(BindingProblems::NotBoundAcrossAll);
                }
            }
            Some(true) => {
                ec.store.bindings[id].problems = Some(BindingProblems::BoundMoreThanOnce);
            }
            Some(false) => {}
        }
        self.is_used.insert(id, true);
    }
}

impl<'db> ExprCollector<'db> {
    pub fn new(
        db: &dyn DefDatabase,
        module: ModuleId,
        current_file_id: HirFileId,
    ) -> ExprCollector<'_> {
        let (def_map, local_def_map) = module.local_def_map(db);
        let expander = Expander::new(db, current_file_id, def_map);
        let krate = module.krate(db);
        let mut result = ExprCollector {
            db,
            cfg_options: krate.cfg_options(db),
            module,
            def_map,
            local_def_map,
            lang_items: OnceCell::new(),
            store: ExpressionStoreBuilder::default(),
            expander,
            current_try_block: None,
            is_lowering_coroutine: false,
            label_ribs: Vec::new(),
            unowned_bindings: Vec::new(),
            awaitable_context: None,
            current_block_legacy_macro_defs_count: FxHashMap::default(),
            outer_impl_trait: false,
            krate,
            name_generator_index: 0,
        };
        result.store.inference_roots = Some(SmallVec::new());
        result
    }

    fn generate_new_name(&mut self) -> Name {
        let index = self.name_generator_index;
        self.name_generator_index += 1;
        Name::generate_new_name(index)
    }

    #[inline]
    pub(crate) fn lang_items(&self) -> &'db LangItems {
        self.lang_items.get_or_init(|| crate::lang_item::lang_items(self.db, self.def_map.krate()))
    }

    #[inline]
    pub(crate) fn span_map(&self) -> SpanMap<'_> {
        self.expander.span_map()
    }

    pub(in crate::expr_store) fn lower_lifetime_ref(
        &mut self,
        lifetime: ast::Lifetime,
    ) -> LifetimeRefId {
        // FIXME: Keyword check?
        let lifetime_ref = match &*lifetime.text() {
            "" | "'" => LifetimeRef::Error,
            "'static" => LifetimeRef::Static,
            "'_" => LifetimeRef::Placeholder,
            text => LifetimeRef::Named(Name::new_lifetime(text)),
        };
        self.alloc_lifetime_ref(lifetime_ref, AstPtr::new(&lifetime))
    }

    pub(in crate::expr_store) fn lower_lifetime_ref_opt(
        &mut self,
        lifetime: Option<ast::Lifetime>,
    ) -> LifetimeRefId {
        match lifetime {
            Some(lifetime) => self.lower_lifetime_ref(lifetime),
            None => self.alloc_lifetime_ref_desugared(LifetimeRef::Placeholder),
        }
    }

    /// Converts an `ast::TypeRef` to a `hir::TypeRef`.
    pub(in crate::expr_store) fn lower_type_ref(
        &mut self,
        node: ast::Type,
        impl_trait_lower_fn: ImplTraitLowerFn<'_>,
    ) -> TypeRefId {
        let ty = match &node {
            ast::Type::ParenType(inner) => {
                return self.lower_type_ref_opt(inner.ty(), impl_trait_lower_fn);
            }
            ast::Type::TupleType(inner) => TypeRef::Tuple(ThinVec::from_iter(Vec::from_iter(
                inner.fields().map(|it| self.lower_type_ref(it, impl_trait_lower_fn)),
            ))),
            ast::Type::NeverType(..) => TypeRef::Never,
            ast::Type::PathType(inner) => inner
                .path()
                .and_then(|it| self.lower_path(it, impl_trait_lower_fn))
                .map(TypeRef::Path)
                .unwrap_or(TypeRef::Error),
            ast::Type::PtrType(inner) => {
                let inner_ty = self.lower_type_ref_opt(inner.ty(), impl_trait_lower_fn);
                let mutability = Mutability::from_mutable(inner.mut_token().is_some());
                TypeRef::RawPtr(inner_ty, mutability)
            }
            ast::Type::ArrayType(inner) => {
                let len = self.lower_const_arg_opt(inner.const_arg());
                TypeRef::Array(ArrayType {
                    ty: self.lower_type_ref_opt(inner.ty(), impl_trait_lower_fn),
                    len,
                })
            }
            ast::Type::SliceType(inner) => {
                TypeRef::Slice(self.lower_type_ref_opt(inner.ty(), impl_trait_lower_fn))
            }
            ast::Type::RefType(inner) => {
                let inner_ty = self.lower_type_ref_opt(inner.ty(), impl_trait_lower_fn);
                let lifetime = inner.lifetime().map(|lt| self.lower_lifetime_ref(lt));
                let mutability = Mutability::from_mutable(inner.mut_token().is_some());
                TypeRef::Reference(Box::new(RefType { ty: inner_ty, lifetime, mutability }))
            }
            ast::Type::InferType(_inner) => TypeRef::Placeholder,
            ast::Type::FnPtrType(inner) => {
                let ret_ty = inner
                    .ret_type()
                    .and_then(|rt| rt.ty())
                    .map(|it| self.lower_type_ref(it, impl_trait_lower_fn))
                    .unwrap_or_else(|| self.alloc_type_ref_desugared(TypeRef::unit()));
                let mut is_varargs = false;
                let mut params = if let Some(pl) = inner.param_list() {
                    if let Some(param) = pl.params().last() {
                        is_varargs = param.dotdotdot_token().is_some();
                    }

                    pl.params()
                        .map(|it| {
                            let type_ref = self.lower_type_ref_opt(it.ty(), impl_trait_lower_fn);
                            let name = match it.pat() {
                                Some(ast::Pat::IdentPat(it)) => Some(
                                    it.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing),
                                ),
                                _ => None,
                            };
                            (name, type_ref)
                        })
                        .collect()
                } else {
                    Vec::with_capacity(1)
                };
                fn lower_abi(abi: ast::Abi) -> ExternAbi {
                    abi.abi_string()
                        .and_then(|abi| abi.text_without_quotes().parse().ok())
                        .unwrap_or(ExternAbi::FALLBACK)
                }

                let abi = inner.abi().map(lower_abi).unwrap_or(ExternAbi::Rust);
                params.push((None, ret_ty));
                TypeRef::Fn(Box::new(FnType {
                    is_varargs,
                    is_unsafe: inner.unsafe_token().is_some(),
                    abi,
                    params: params.into_boxed_slice(),
                }))
            }
            // for types are close enough for our purposes to the inner type for now...
            ast::Type::ForType(inner) => {
                return self.lower_type_ref_opt(inner.ty(), impl_trait_lower_fn);
            }
            ast::Type::ImplTraitType(inner) => {
                if self.outer_impl_trait {
                    // Disallow nested impl traits
                    TypeRef::Error
                } else {
                    return self.with_outer_impl_trait_scope(true, |this| {
                        let type_bounds =
                            this.type_bounds_from_ast(inner.type_bound_list(), impl_trait_lower_fn);
                        impl_trait_lower_fn(this, AstPtr::new(&node), type_bounds)
                    });
                }
            }
            ast::Type::DynTraitType(inner) => TypeRef::DynTrait(
                self.type_bounds_from_ast(inner.type_bound_list(), impl_trait_lower_fn),
            ),
            ast::Type::PatternType(inner) => TypeRef::PatternType(
                self.lower_type_ref_opt(inner.ty(), impl_trait_lower_fn),
                self.collect_ty_pat_opt(inner.pat()),
            ),
            ast::Type::MacroType(mt) => match mt.macro_call() {
                Some(mcall) => {
                    let macro_ptr = AstPtr::new(&mcall);
                    let src = self.expander.in_file(AstPtr::new(&node));
                    let id = self.collect_macro_call(mcall, macro_ptr, true, |this, expansion| {
                        this.lower_type_ref_opt(expansion, impl_trait_lower_fn)
                    });
                    self.store.types_map.insert(src, id);
                    return id;
                }
                None => TypeRef::Error,
            },
        };
        self.alloc_type_ref(ty, AstPtr::new(&node))
    }

    pub(crate) fn lower_type_ref_disallow_impl_trait(&mut self, node: ast::Type) -> TypeRefId {
        self.lower_type_ref(node, &mut Self::impl_trait_error_allocator)
    }

    pub(crate) fn lower_type_ref_opt(
        &mut self,
        node: Option<ast::Type>,
        impl_trait_lower_fn: ImplTraitLowerFn<'_>,
    ) -> TypeRefId {
        match node {
            Some(node) => self.lower_type_ref(node, impl_trait_lower_fn),
            None => self.alloc_error_type(),
        }
    }

    pub(crate) fn lower_type_ref_opt_disallow_impl_trait(
        &mut self,
        node: Option<ast::Type>,
    ) -> TypeRefId {
        self.lower_type_ref_opt(node, &mut Self::impl_trait_error_allocator)
    }

    fn alloc_type_ref(&mut self, type_ref: TypeRef, node: TypePtr) -> TypeRefId {
        let id = self.store.types.alloc(type_ref);
        let ptr = self.expander.in_file(node);
        self.store.types_map_back.insert(id, ptr);
        self.store.types_map.insert(ptr, id);
        id
    }

    fn alloc_lifetime_ref(
        &mut self,
        lifetime_ref: LifetimeRef,
        node: LifetimePtr,
    ) -> LifetimeRefId {
        let id = self.store.lifetimes.alloc(lifetime_ref);
        let ptr = self.expander.in_file(node);
        self.store.lifetime_map_back.insert(id, ptr);
        self.store.lifetime_map.insert(ptr, id);
        id
    }

    fn alloc_type_ref_desugared(&mut self, type_ref: TypeRef) -> TypeRefId {
        self.store.types.alloc(type_ref)
    }

    fn alloc_lifetime_ref_desugared(&mut self, lifetime_ref: LifetimeRef) -> LifetimeRefId {
        self.store.lifetimes.alloc(lifetime_ref)
    }

    fn alloc_error_type(&mut self) -> TypeRefId {
        self.store.types.alloc(TypeRef::Error)
    }

    pub fn lower_path(
        &mut self,
        ast: ast::Path,
        impl_trait_lower_fn: ImplTraitLowerFn<'_>,
    ) -> Option<Path> {
        super::lower::path::lower_path(self, ast, impl_trait_lower_fn)
    }

    fn with_outer_impl_trait_scope<R>(
        &mut self,
        impl_trait: bool,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let old = mem::replace(&mut self.outer_impl_trait, impl_trait);
        let result = f(self);
        self.outer_impl_trait = old;
        result
    }

    pub fn impl_trait_error_allocator(
        ec: &mut ExprCollector<'_>,
        ptr: TypePtr,
        _: ThinVec<TypeBound>,
    ) -> TypeRefId {
        ec.alloc_type_ref(TypeRef::Error, ptr)
    }

    fn impl_trait_allocator(
        ec: &mut ExprCollector<'_>,
        ptr: TypePtr,
        bounds: ThinVec<TypeBound>,
    ) -> TypeRefId {
        ec.alloc_type_ref(TypeRef::ImplTrait(bounds), ptr)
    }

    fn alloc_path(&mut self, path: Path, node: TypePtr) -> PathId {
        PathId::from_type_ref_unchecked(self.alloc_type_ref(TypeRef::Path(path), node))
    }

    /// Collect `GenericArgs` from the parts of a fn-like path, i.e. `Fn(X, Y)
    /// -> Z` (which desugars to `Fn<(X, Y), Output=Z>`).
    pub(in crate::expr_store) fn lower_generic_args_from_fn_path(
        &mut self,
        args: Option<ast::ParenthesizedArgList>,
        ret_type: Option<ast::RetType>,
        impl_trait_lower_fn: ImplTraitLowerFn<'_>,
    ) -> Option<GenericArgs> {
        let params = args?;
        let mut param_types = Vec::new();
        for param in params.type_args() {
            let type_ref = self.lower_type_ref_opt(param.ty(), impl_trait_lower_fn);
            param_types.push(type_ref);
        }
        let args = Box::new([GenericArg::Type(
            self.alloc_type_ref_desugared(TypeRef::Tuple(ThinVec::from_iter(param_types))),
        )]);
        let bindings = if let Some(ret_type) = ret_type {
            let type_ref = self.lower_type_ref_opt(ret_type.ty(), impl_trait_lower_fn);
            Box::new([AssociatedTypeBinding {
                name: Name::new_symbol_root(sym::Output),
                args: None,
                type_ref: Some(type_ref),
                bounds: Box::default(),
            }])
        } else {
            // -> ()
            let type_ref = self.alloc_type_ref_desugared(TypeRef::unit());
            Box::new([AssociatedTypeBinding {
                name: Name::new_symbol_root(sym::Output),
                args: None,
                type_ref: Some(type_ref),
                bounds: Box::default(),
            }])
        };
        Some(GenericArgs {
            args,
            has_self_type: false,
            bindings,
            parenthesized: GenericArgsParentheses::ParenSugar,
        })
    }

    pub(super) fn lower_generic_args(
        &mut self,
        node: ast::GenericArgList,
        impl_trait_lower_fn: ImplTraitLowerFn<'_>,
    ) -> Option<GenericArgs> {
        // This needs to be kept in sync with `hir_generic_arg_to_ast()`.
        let mut args = Vec::new();
        let mut bindings = Vec::new();
        for generic_arg in node.generic_args() {
            match generic_arg {
                ast::GenericArg::TypeArg(type_arg) => {
                    let type_ref = self.lower_type_ref_opt(type_arg.ty(), impl_trait_lower_fn);
                    args.push(GenericArg::Type(type_ref));
                }
                ast::GenericArg::AssocTypeArg(assoc_type_arg) => {
                    // This needs to be kept in sync with `hir_assoc_type_binding_to_ast()`.
                    if assoc_type_arg.param_list().is_some() {
                        // We currently ignore associated return type bounds.
                        continue;
                    }
                    if let Some(name_ref) = assoc_type_arg.name_ref() {
                        // Nested impl traits like `impl Foo<Assoc = impl Bar>` are allowed
                        self.with_outer_impl_trait_scope(false, |this| {
                            let name = name_ref.as_name();
                            let args = assoc_type_arg
                                .generic_arg_list()
                                .and_then(|args| this.lower_generic_args(args, impl_trait_lower_fn))
                                .or_else(|| {
                                    assoc_type_arg
                                        .return_type_syntax()
                                        .map(|_| GenericArgs::return_type_notation())
                                });
                            let type_ref = assoc_type_arg
                                .ty()
                                .map(|it| this.lower_type_ref(it, impl_trait_lower_fn));
                            let bounds = if let Some(l) = assoc_type_arg.type_bound_list() {
                                l.bounds()
                                    .map(|it| this.lower_type_bound(it, impl_trait_lower_fn))
                                    .collect()
                            } else {
                                Box::default()
                            };
                            bindings.push(AssociatedTypeBinding { name, args, type_ref, bounds });
                        });
                    }
                }
                ast::GenericArg::LifetimeArg(lifetime_arg) => {
                    if let Some(lifetime) = lifetime_arg.lifetime() {
                        let lifetime_ref = self.lower_lifetime_ref(lifetime);
                        args.push(GenericArg::Lifetime(lifetime_ref))
                    }
                }
                ast::GenericArg::ConstArg(arg) => {
                    let arg = self.lower_const_arg(arg);
                    args.push(GenericArg::Const(arg))
                }
            }
        }

        if args.is_empty() && bindings.is_empty() {
            return None;
        }
        Some(GenericArgs {
            args: args.into_boxed_slice(),
            has_self_type: false,
            bindings: bindings.into_boxed_slice(),
            parenthesized: GenericArgsParentheses::No,
        })
    }

    /// Lowers a desugared coroutine body after moving all of the arguments
    /// into the body. This is to make sure that the future actually owns the
    /// arguments that are passed to the function, and to ensure things like
    /// drop order are stable.
    fn lower_coroutine_body_with_moved_arguments(
        &mut self,
        self_params: &mut ArrayVec<BindingId, 2>,
        params: &mut [PatId],
        body: ExprId,
        kind: CoroutineKind,
        coroutine_source: CoroutineSource,
    ) -> ExprId {
        // Async function parameters are lowered into the closure body so that they are
        // captured and so that the drop order matches the equivalent non-async functions.
        //
        // from:
        //
        //     async fn foo(<pattern>: <ty>, <pattern>: <ty>, <pattern>: <ty>) {
        //         <body>
        //     }
        //
        // into:
        //
        //     fn foo(__arg0: <ty>, __arg1: <ty>, __arg2: <ty>) {
        //       async move {
        //         let __arg2 = __arg2;
        //         let <pattern> = __arg2;
        //         let __arg1 = __arg1;
        //         let <pattern> = __arg1;
        //         let __arg0 = __arg0;
        //         let <pattern> = __arg0;
        //         drop-temps { <body> } // see comments later in fn for details
        //       }
        //     }
        //
        // If `<pattern>` is a simple ident, then it is lowered to a single
        // `let <pattern> = <pattern>;` statement as an optimization.

        let mut statements = Vec::new();

        if let Some(&self_param) = self_params.first() {
            let Binding { ref name, mode, hygiene, .. } = self.store.bindings[self_param];
            let name = name.clone();
            let child_binding_id = self.alloc_binding(name.clone(), mode, hygiene);
            let child_pat_id =
                self.alloc_pat_desugared(Pat::Bind { id: child_binding_id, subpat: None });
            self.add_definition_to_binding(child_binding_id, child_pat_id);
            let expr = self.alloc_expr_desugared(Expr::Path(name.into()));
            if !hygiene.is_root() {
                self.store.ident_hygiene.insert(expr.into(), hygiene);
            }
            statements.push(Statement::Let {
                pat: child_pat_id,
                type_ref: None,
                initializer: Some(expr),
                else_branch: None,
            });
            self_params.push(child_binding_id);
        }

        for param in params {
            let (name, hygiene, is_simple_parameter) = match self.store.pats[*param] {
                // Check if this is a binding pattern, if so, we can optimize and avoid adding a
                // `let <pat> = __argN;` statement. In this case, we do not rename the parameter.
                Pat::Bind { id, subpat: None, .. }
                    if matches!(
                        self.store.bindings[id].mode,
                        BindingAnnotation::Unannotated | BindingAnnotation::Mutable
                    ) =>
                {
                    (self.store.bindings[id].name.clone(), self.store.bindings[id].hygiene, true)
                }
                Pat::Bind { id, .. } => {
                    // If this is a `ref` binding, we can't leave it as is but we can at least reuse the name, for better display.
                    (self.store.bindings[id].name.clone(), self.store.bindings[id].hygiene, false)
                }
                _ => (self.generate_new_name(), HygieneId::ROOT, false),
            };
            let pat_syntax = self.store.pat_map_back.get(*param).copied();
            let child_binding_id =
                self.alloc_binding(name.clone(), BindingAnnotation::Mutable, hygiene);
            let child_pat_id =
                self.alloc_pat_desugared(Pat::Bind { id: child_binding_id, subpat: None });
            self.add_definition_to_binding(child_binding_id, child_pat_id);
            if let Some(pat_syntax) = pat_syntax {
                self.store.pat_map_back.insert(child_pat_id, pat_syntax);
            }
            let expr = self.alloc_expr_desugared(Expr::Path(name.clone().into()));
            if !hygiene.is_root() {
                self.store.ident_hygiene.insert(expr.into(), hygiene);
            }
            statements.push(Statement::Let {
                pat: child_pat_id,
                type_ref: None,
                initializer: Some(expr),
                else_branch: None,
            });
            if !is_simple_parameter {
                let expr = self.alloc_expr_desugared(Expr::Path(name.clone().into()));
                if !hygiene.is_root() {
                    self.store.ident_hygiene.insert(expr.into(), hygiene);
                }
                statements.push(Statement::Let {
                    pat: *param,
                    type_ref: None,
                    initializer: Some(expr),
                    else_branch: None,
                });

                let parent_binding_id =
                    self.alloc_binding(name.clone(), BindingAnnotation::Mutable, hygiene);
                let parent_pat_id =
                    self.alloc_pat_desugared(Pat::Bind { id: parent_binding_id, subpat: None });
                self.add_definition_to_binding(parent_binding_id, parent_pat_id);
                if let Some(pat_syntax) = pat_syntax {
                    self.store.pat_map_back.insert(parent_pat_id, pat_syntax);
                }
                *param = parent_pat_id;
            }
        }

        let coroutine = self.desugared_coroutine_expr(
            kind,
            coroutine_source,
            // The default capture mode here is by-ref. Later on during upvar analysis,
            // we will force the captured arguments to by-move, but for async closures,
            // we want to make sure that we avoid unnecessarily moving captures, or else
            // all async closures would default to `FnOnce` as their calling mode.
            CaptureBy::Ref,
            None,
            statements.into_boxed_slice(),
            Some(body),
        );
        // It's important that this comes last, see the lowering of async closures for why.
        self.alloc_expr_desugared(coroutine)
    }

    fn desugared_coroutine_expr(
        &mut self,
        kind: CoroutineKind,
        source: CoroutineSource,
        capture_by: CaptureBy,
        id: Option<BlockId>,
        statements: Box<[Statement]>,
        tail: Option<ExprId>,
    ) -> Expr {
        let block = self.alloc_expr_desugared(Expr::Block { label: None, id, statements, tail });
        Expr::Closure {
            args: Box::default(),
            arg_types: Box::default(),
            ret_type: None,
            body: block,
            closure_kind: ClosureKind::Coroutine { kind, source },
            capture_by,
        }
    }

    fn collect(
        &mut self,
        self_params: &mut ArrayVec<BindingId, 2>,
        params: &mut [PatId],
        expr: Option<ast::Expr>,
        awaitable: Awaitable,
        is_async_fn: bool,
        is_gen_fn: bool,
    ) -> ExprId {
        self.awaitable_context.replace(awaitable);
        self.with_label_rib(RibKind::Closure, |this| {
            let body = this.collect_expr_opt(expr);
            if is_async_fn || is_gen_fn {
                let kind = match (is_async_fn, is_gen_fn) {
                    (true, true) => CoroutineKind::AsyncGen,
                    (true, false) => CoroutineKind::Async,
                    (false, true) => CoroutineKind::Gen,
                    (false, false) => unreachable!(),
                };
                this.lower_coroutine_body_with_moved_arguments(
                    self_params,
                    params,
                    body,
                    kind,
                    CoroutineSource::Fn,
                )
            } else {
                body
            }
        })
    }

    fn type_bounds_from_ast(
        &mut self,
        type_bounds_opt: Option<ast::TypeBoundList>,
        impl_trait_lower_fn: ImplTraitLowerFn<'_>,
    ) -> ThinVec<TypeBound> {
        if let Some(type_bounds) = type_bounds_opt {
            ThinVec::from_iter(Vec::from_iter(
                type_bounds.bounds().map(|it| self.lower_type_bound(it, impl_trait_lower_fn)),
            ))
        } else {
            ThinVec::from_iter([])
        }
    }

    fn lower_path_type(
        &mut self,
        path_type: &ast::PathType,
        impl_trait_lower_fn: ImplTraitLowerFn<'_>,
    ) -> Option<Path> {
        let path = self.lower_path(path_type.path()?, impl_trait_lower_fn)?;
        Some(path)
    }

    fn lower_type_bound(
        &mut self,
        node: ast::TypeBound,
        impl_trait_lower_fn: ImplTraitLowerFn<'_>,
    ) -> TypeBound {
        let Some(kind) = node.kind() else { return TypeBound::Error };
        match kind {
            ast::TypeBoundKind::PathType(binder, path_type) => {
                let binder = match binder.and_then(|it| it.generic_param_list()) {
                    Some(gpl) => gpl
                        .lifetime_params()
                        .flat_map(|lp| lp.lifetime().map(|lt| Name::new_lifetime(&lt.text())))
                        .collect(),
                    None => ThinVec::default(),
                };
                let m = match node.question_mark_token() {
                    Some(_) => TraitBoundModifier::Maybe,
                    None => TraitBoundModifier::None,
                };
                self.lower_path_type(&path_type, impl_trait_lower_fn)
                    .map(|p| {
                        let path = self.alloc_path(p, AstPtr::new(&path_type).upcast());
                        if binder.is_empty() {
                            TypeBound::Path(path, m)
                        } else {
                            TypeBound::ForLifetime(binder, path)
                        }
                    })
                    .unwrap_or(TypeBound::Error)
            }
            ast::TypeBoundKind::Use(gal) => TypeBound::Use(
                gal.use_bound_generic_args()
                    .map(|p| match p {
                        ast::UseBoundGenericArg::Lifetime(l) => {
                            UseArgRef::Lifetime(self.lower_lifetime_ref(l))
                        }
                        ast::UseBoundGenericArg::NameRef(n) => UseArgRef::Name(n.as_name()),
                    })
                    .collect(),
            ),
            ast::TypeBoundKind::Lifetime(lifetime) => {
                TypeBound::Lifetime(self.lower_lifetime_ref(lifetime))
            }
        }
    }

    fn lower_const_arg_opt(&mut self, arg: Option<ast::ConstArg>) -> ConstRef {
        ConstRef {
            expr: self.with_fresh_binding_expr_root(|this| {
                this.collect_expr_opt(arg.and_then(|arg| arg.expr()))
            }),
        }
    }

    pub fn lower_const_arg(&mut self, arg: ast::ConstArg) -> ConstRef {
        ConstRef {
            expr: self.with_fresh_binding_expr_root(|this| this.collect_expr_opt(arg.expr())),
        }
    }

    fn collect_expr(&mut self, expr: ast::Expr) -> ExprId {
        self.maybe_collect_expr(expr).unwrap_or_else(|| self.missing_expr())
    }

    pub(in crate::expr_store) fn collect_expr_opt(&mut self, expr: Option<ast::Expr>) -> ExprId {
        match expr {
            Some(expr) => self.collect_expr(expr),
            None => self.missing_expr(),
        }
    }

    /// Returns `None` if and only if the expression is `#[cfg]`d out.
    fn maybe_collect_expr(&mut self, expr: ast::Expr) -> Option<ExprId> {
        let syntax_ptr = AstPtr::new(&expr);
        if !self.check_cfg(&expr) {
            return None;
        }

        // FIXME: Move some of these arms out into separate methods for clarity
        Some(match expr {
            ast::Expr::IfExpr(e) => {
                let then_branch = self.collect_block_opt(e.then_branch());

                let else_branch = e.else_branch().map(|b| match b {
                    ast::ElseBranch::Block(it) => self.collect_block(it),
                    ast::ElseBranch::IfExpr(elif) => {
                        let expr: ast::Expr = ast::Expr::cast(elif.syntax().clone()).unwrap();
                        self.collect_expr(expr)
                    }
                });

                let condition = self.collect_expr_opt(e.condition());

                self.alloc_expr(Expr::If { condition, then_branch, else_branch }, syntax_ptr)
            }
            ast::Expr::LetExpr(e) => {
                let pat = self.collect_pat_top(e.pat());
                let expr = self.collect_expr_opt(e.expr());
                self.alloc_expr(Expr::Let { pat, expr }, syntax_ptr)
            }
            ast::Expr::BlockExpr(e) => match e.modifier() {
                Some(ast::BlockModifier::Try { try_token: _, bikeshed_token: _, result_type }) => {
                    self.desugar_try_block(e, result_type)
                }
                Some(ast::BlockModifier::Unsafe(_)) => {
                    self.collect_block_(e, |_, id, statements, tail| Expr::Unsafe {
                        id,
                        statements,
                        tail,
                    })
                }
                Some(ast::BlockModifier::Label(label)) => {
                    let label_hygiene = self.hygiene_id_for(label.syntax().text_range());
                    let label_id = self.collect_label(label);
                    self.with_labeled_rib(label_id, label_hygiene, |this| {
                        this.collect_block_(e, |_, id, statements, tail| Expr::Block {
                            id,
                            statements,
                            tail,
                            label: Some(label_id),
                        })
                    })
                }
                Some(ast::BlockModifier::Async(_)) => {
                    let capture_by =
                        if e.move_token().is_some() { CaptureBy::Value } else { CaptureBy::Ref };
                    self.with_label_rib(RibKind::Closure, |this| {
                        this.with_awaitable_block(Awaitable::Yes, |this| {
                            this.collect_block_(e, |this, id, statements, tail| {
                                this.desugared_coroutine_expr(
                                    CoroutineKind::Async,
                                    CoroutineSource::Block,
                                    capture_by,
                                    id,
                                    statements,
                                    tail,
                                )
                            })
                        })
                    })
                }
                Some(ast::BlockModifier::Gen(_)) => {
                    let capture_by =
                        if e.move_token().is_some() { CaptureBy::Value } else { CaptureBy::Ref };
                    self.with_label_rib(RibKind::Closure, |this| {
                        this.with_awaitable_block(Awaitable::No("non-async gen block"), |this| {
                            this.collect_block_(e, |this, id, statements, tail| {
                                this.desugared_coroutine_expr(
                                    CoroutineKind::Gen,
                                    CoroutineSource::Block,
                                    capture_by,
                                    id,
                                    statements,
                                    tail,
                                )
                            })
                        })
                    })
                }
                Some(ast::BlockModifier::AsyncGen(_)) => {
                    let capture_by =
                        if e.move_token().is_some() { CaptureBy::Value } else { CaptureBy::Ref };
                    self.with_label_rib(RibKind::Closure, |this| {
                        this.with_awaitable_block(Awaitable::Yes, |this| {
                            this.collect_block_(e, |this, id, statements, tail| {
                                this.desugared_coroutine_expr(
                                    CoroutineKind::AsyncGen,
                                    CoroutineSource::Block,
                                    capture_by,
                                    id,
                                    statements,
                                    tail,
                                )
                            })
                        })
                    })
                }
                Some(ast::BlockModifier::Const(_)) => {
                    self.with_label_rib(RibKind::Constant, |this| {
                        this.with_awaitable_block(Awaitable::No("constant block"), |this| {
                            this.with_binding_owner(|this| {
                                let inner_expr = this.collect_block(e);
                                this.alloc_expr(Expr::Const(inner_expr), syntax_ptr)
                            })
                        })
                    })
                }
                None => self.collect_block(e),
            },
            ast::Expr::LoopExpr(e) => {
                let label = e.label().map(|label| {
                    (self.hygiene_id_for(label.syntax().text_range()), self.collect_label(label))
                });
                let body = self.collect_labelled_block_opt(label, e.loop_body());
                self.alloc_expr(Expr::Loop { body, label: label.map(|it| it.1) }, syntax_ptr)
            }
            ast::Expr::WhileExpr(e) => self.collect_while_loop(syntax_ptr, e),
            ast::Expr::ForExpr(e) => self.collect_for_loop(syntax_ptr, e),
            ast::Expr::CallExpr(e) => {
                // FIXME(MINIMUM_SUPPORTED_TOOLCHAIN_VERSION): Remove this once we drop support for <1.86, https://github.com/rust-lang/rust/commit/ac9cb908ac4301dfc25e7a2edee574320022ae2c
                let is_rustc_box = {
                    let attrs = e.attrs();
                    attrs.filter_map(|it| it.as_simple_atom()).any(|it| it == "rustc_box")
                };
                if is_rustc_box {
                    let expr = self.collect_expr_opt(e.arg_list().and_then(|it| it.args().next()));
                    self.alloc_expr(Expr::Box { expr }, syntax_ptr)
                } else {
                    let callee = self.collect_expr_opt(e.expr());
                    let args = if let Some(arg_list) = e.arg_list() {
                        arg_list.args().filter_map(|e| self.maybe_collect_expr(e)).collect()
                    } else {
                        Box::default()
                    };
                    self.alloc_expr(Expr::Call { callee, args }, syntax_ptr)
                }
            }
            ast::Expr::MethodCallExpr(e) => {
                let receiver = self.collect_expr_opt(e.receiver());
                let args = if let Some(arg_list) = e.arg_list() {
                    arg_list.args().filter_map(|e| self.maybe_collect_expr(e)).collect()
                } else {
                    Box::default()
                };
                let method_name = e.name_ref().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                let generic_args = e
                    .generic_arg_list()
                    .and_then(|it| {
                        self.lower_generic_args(it, &mut Self::impl_trait_error_allocator)
                    })
                    .map(Box::new);
                self.alloc_expr(
                    Expr::MethodCall { receiver, method_name, args, generic_args },
                    syntax_ptr,
                )
            }
            ast::Expr::MatchExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let arms = if let Some(match_arm_list) = e.match_arm_list() {
                    match_arm_list
                        .arms()
                        .filter_map(|arm| {
                            if self.check_cfg(&arm) {
                                Some(MatchArm {
                                    pat: self.collect_pat_top(arm.pat()),
                                    expr: self.collect_expr_opt(arm.expr()),
                                    guard: arm
                                        .guard()
                                        .map(|guard| self.collect_expr_opt(guard.condition())),
                                })
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    Box::default()
                };
                self.alloc_expr(Expr::Match { expr, arms }, syntax_ptr)
            }
            ast::Expr::PathExpr(e) => {
                let (path, hygiene) = self
                    .collect_expr_path(e)
                    .map(|(path, hygiene)| (Expr::Path(path), hygiene))
                    .unwrap_or((Expr::Missing, HygieneId::ROOT));
                let expr_id = self.alloc_expr(path, syntax_ptr);
                if !hygiene.is_root() {
                    self.store.ident_hygiene.insert(expr_id.into(), hygiene);
                }
                expr_id
            }
            ast::Expr::ContinueExpr(e) => {
                let label = self.resolve_label(e.lifetime()).unwrap_or_else(|e| {
                    self.store.diagnostics.push(e);
                    None
                });
                self.alloc_expr(Expr::Continue { label }, syntax_ptr)
            }
            ast::Expr::BreakExpr(e) => {
                let label = self.resolve_label(e.lifetime()).unwrap_or_else(|e| {
                    self.store.diagnostics.push(e);
                    None
                });
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Break { expr, label }, syntax_ptr)
            }
            ast::Expr::ParenExpr(e) => {
                let inner = self.collect_expr_opt(e.expr());
                // make the paren expr point to the inner expression as well for IDE resolution
                let src = self.expander.in_file(syntax_ptr);
                self.store.expr_map.insert(src, inner.into());
                inner
            }
            ast::Expr::ReturnExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Return { expr }, syntax_ptr)
            }
            ast::Expr::BecomeExpr(e) => {
                let expr =
                    e.expr().map(|e| self.collect_expr(e)).unwrap_or_else(|| self.missing_expr());
                self.alloc_expr(Expr::Become { expr }, syntax_ptr)
            }
            ast::Expr::YieldExpr(e) => {
                self.is_lowering_coroutine = true;
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Yield { expr }, syntax_ptr)
            }
            ast::Expr::YeetExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Yeet { expr }, syntax_ptr)
            }
            ast::Expr::RecordExpr(e) => {
                let path = e
                    .path()
                    .and_then(|path| self.lower_path(path, &mut Self::impl_trait_error_allocator));
                let Some(path) = path else {
                    return Some(self.missing_expr());
                };
                let record_lit = if let Some(nfl) = e.record_expr_field_list() {
                    let fields = nfl
                        .fields()
                        .filter_map(|field| {
                            if !self.check_cfg(&field) {
                                return None;
                            }

                            let name = field.field_name()?.as_name();

                            let expr = match field.expr() {
                                Some(e) => self.collect_expr(e),
                                None => self.missing_expr(),
                            };
                            let src = self.expander.in_file(AstPtr::new(&field));
                            self.store.field_map_back.insert(expr, src);
                            Some(RecordLitField { name, expr })
                        })
                        .collect();
                    let spread_expr = nfl.spread().map(|s| self.collect_expr(s));
                    let has_spread_syntax = nfl.dotdot_token().is_some();
                    let spread = match (spread_expr, has_spread_syntax) {
                        (None, false) => RecordSpread::None,
                        (None, true) => RecordSpread::FieldDefaults,
                        (Some(expr), _) => RecordSpread::Expr(expr),
                    };
                    Expr::RecordLit { path, fields, spread }
                } else {
                    Expr::RecordLit { path, fields: Box::default(), spread: RecordSpread::None }
                };

                self.alloc_expr(record_lit, syntax_ptr)
            }
            ast::Expr::FieldExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let name = match e.field_access() {
                    Some(kind) => kind.as_name(),
                    _ => Name::missing(),
                };
                self.alloc_expr(Expr::Field { expr, name }, syntax_ptr)
            }
            ast::Expr::AwaitExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                if let Awaitable::No(location) = self.is_lowering_awaitable_block() {
                    self.store.diagnostics.push(ExpressionStoreDiagnostics::AwaitOutsideOfAsync {
                        node: self.expander.in_file(AstPtr::new(&e)),
                        location: location.to_string(),
                    });
                }
                self.alloc_expr(Expr::Await { expr }, syntax_ptr)
            }
            ast::Expr::TryExpr(e) => self.collect_try_operator(syntax_ptr, e),
            ast::Expr::CastExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let type_ref = self.lower_type_ref_opt_disallow_impl_trait(e.ty());
                self.alloc_expr(Expr::Cast { expr, type_ref }, syntax_ptr)
            }
            ast::Expr::RefExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let raw_tok = e.raw_token().is_some();
                let mutability = if raw_tok {
                    if e.mut_token().is_some() { Mutability::Mut } else { Mutability::Shared }
                } else {
                    Mutability::from_mutable(e.mut_token().is_some())
                };
                let rawness = Rawness::from_raw(raw_tok);
                self.alloc_expr(Expr::Ref { expr, rawness, mutability }, syntax_ptr)
            }
            ast::Expr::PrefixExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                match e.op_kind() {
                    Some(op) => self.alloc_expr(Expr::UnaryOp { expr, op }, syntax_ptr),
                    None => self.alloc_expr(Expr::Missing, syntax_ptr),
                }
            }
            ast::Expr::ClosureExpr(e) => self.with_label_rib(RibKind::Closure, |this| {
                let mut is_coroutine_closure = false;
                let closure = this.with_binding_owner_and_return(|this| {
                    let mut args = Vec::new();
                    let mut arg_types = Vec::new();
                    // For coroutine closures, the body, aka. the coroutine is the bindings owner, and not the closure.
                    if let Some(pl) = e.param_list() {
                        let num_params = pl.params().count();
                        args.reserve_exact(num_params);
                        arg_types.reserve_exact(num_params);
                        for param in pl.params() {
                            let pat = this.collect_pat_top(param.pat());
                            let type_ref =
                                param.ty().map(|it| this.lower_type_ref_disallow_impl_trait(it));
                            args.push(pat);
                            arg_types.push(type_ref);
                        }
                    }
                    let ret_type = e
                        .ret_type()
                        .and_then(|r| r.ty())
                        .map(|it| this.lower_type_ref_disallow_impl_trait(it));

                    let prev_is_lowering_coroutine = mem::take(&mut this.is_lowering_coroutine);
                    let prev_try_block = this.current_try_block.take();

                    let awaitable = if e.async_token().is_some() {
                        Awaitable::Yes
                    } else {
                        Awaitable::No("non-async closure")
                    };
                    let mut body = this
                        .with_awaitable_block(awaitable, |this| this.collect_expr_opt(e.body()));
                    let kind = {
                        if e.async_token().is_some() && e.gen_token().is_some() {
                            Some(CoroutineKind::AsyncGen)
                        } else if e.async_token().is_some() {
                            Some(CoroutineKind::Async)
                        } else if e.gen_token().is_some() {
                            Some(CoroutineKind::Gen)
                        } else {
                            None
                        }
                    };

                    let closure_kind = if let Some(kind) = kind {
                        // It's important that this expr is allocated immediately before the closure.
                        // We rely on it for `coroutine_for_closure()`.
                        body = this.lower_coroutine_body_with_moved_arguments(
                            &mut ArrayVec::new(),
                            &mut args,
                            body,
                            kind,
                            CoroutineSource::Closure,
                        );
                        is_coroutine_closure = true;

                        ClosureKind::CoroutineClosure(kind)
                    } else if this.is_lowering_coroutine {
                        let movability = if e.static_token().is_some() {
                            Movability::Static
                        } else {
                            Movability::Movable
                        };
                        ClosureKind::OldCoroutine(movability)
                    } else {
                        ClosureKind::Closure
                    };
                    let capture_by =
                        if e.move_token().is_some() { CaptureBy::Value } else { CaptureBy::Ref };
                    this.is_lowering_coroutine = prev_is_lowering_coroutine;
                    this.current_try_block = prev_try_block;
                    let closure = this.alloc_expr(
                        Expr::Closure {
                            args: args.into(),
                            arg_types: arg_types.into(),
                            ret_type,
                            body,
                            closure_kind,
                            capture_by,
                        },
                        syntax_ptr,
                    );

                    (if is_coroutine_closure { body } else { closure }, closure)
                });

                if is_coroutine_closure {
                    let Expr::Closure { args, .. } = &this.store.exprs[closure] else {
                        unreachable!()
                    };
                    for &arg in args {
                        let Pat::Bind { id, .. } = this.store.pats[arg] else {
                            never!("`lower_coroutine_body_with_moved_arguments()` should make sure the coroutine closure only have simple bind args");
                            continue;
                        };
                        this.store.binding_owners.insert(id, closure);
                    }
                }

                closure
            }),
            ast::Expr::BinExpr(e) => {
                let op = e.op_kind();
                if let Some(ast::BinaryOp::Assignment { op: None }) = op {
                    let target = self.collect_expr_as_pat_opt(e.lhs());
                    let value = self.collect_expr_opt(e.rhs());
                    self.alloc_expr(Expr::Assignment { target, value }, syntax_ptr)
                } else {
                    let lhs = self.collect_expr_opt(e.lhs());
                    let rhs = self.collect_expr_opt(e.rhs());
                    self.alloc_expr(Expr::BinaryOp { lhs, rhs, op }, syntax_ptr)
                }
            }
            ast::Expr::TupleExpr(e) => {
                let mut exprs: Vec<_> = e.fields().map(|expr| self.collect_expr(expr)).collect();
                // if there is a leading comma, the user is most likely to type out a leading expression
                // so we insert a missing expression at the beginning for IDE features
                if comma_follows_token(e.l_paren_token()) {
                    exprs.insert(0, self.missing_expr());
                }

                self.alloc_expr(Expr::Tuple { exprs: exprs.into_boxed_slice() }, syntax_ptr)
            }
            ast::Expr::ArrayExpr(e) => {
                let kind = e.kind();

                match kind {
                    ArrayExprKind::ElementList(e) => {
                        let elements = e
                            .filter_map(|expr| {
                                if self.check_cfg(&expr) {
                                    Some(self.collect_expr(expr))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        self.alloc_expr(Expr::Array(Array::ElementList { elements }), syntax_ptr)
                    }
                    ArrayExprKind::Repeat { initializer, repeat } => {
                        let initializer = self.collect_expr_opt(initializer);
                        let repeat = self.with_label_rib(RibKind::Constant, |this| {
                            if let Some(repeat) = repeat {
                                this.with_binding_owner(|this| this.collect_expr(repeat))
                            } else {
                                this.missing_expr()
                            }
                        });
                        self.alloc_expr(
                            Expr::Array(Array::Repeat { initializer, repeat }),
                            syntax_ptr,
                        )
                    }
                }
            }

            ast::Expr::Literal(e) => self.alloc_expr(Expr::Literal(e.kind().into()), syntax_ptr),
            ast::Expr::IndexExpr(e) => {
                let base = self.collect_expr_opt(e.base());
                let index = self.collect_expr_opt(e.index());
                self.alloc_expr(Expr::Index { base, index }, syntax_ptr)
            }
            ast::Expr::RangeExpr(e) => {
                let lhs = e.start().map(|lhs| self.collect_expr(lhs));
                let rhs = e.end().map(|rhs| self.collect_expr(rhs));
                match e.op_kind() {
                    Some(range_type) => {
                        self.alloc_expr(Expr::Range { lhs, rhs, range_type }, syntax_ptr)
                    }
                    None => self.alloc_expr(Expr::Missing, syntax_ptr),
                }
            }
            ast::Expr::MacroExpr(e) => {
                let e = e.macro_call()?;
                let macro_ptr = AstPtr::new(&e);
                let id = self.collect_macro_call(e, macro_ptr, true, |this, expansion| {
                    expansion.map(|it| this.collect_expr(it))
                });
                match id {
                    Some(id) => {
                        // Make the macro-call point to its expanded expression so we can query
                        // semantics on syntax pointers to the macro
                        let src = self.expander.in_file(syntax_ptr);
                        self.store.expr_map.insert(src, id.into());
                        id
                    }
                    None => self.alloc_expr(Expr::Missing, syntax_ptr),
                }
            }
            ast::Expr::UnderscoreExpr(_) => self.alloc_expr(Expr::Underscore, syntax_ptr),
            ast::Expr::AsmExpr(e) => self.lower_inline_asm(e, syntax_ptr),
            ast::Expr::OffsetOfExpr(e) => {
                let container = self.lower_type_ref_opt_disallow_impl_trait(e.ty());
                let fields = e.fields().map(|it| it.as_name()).collect();
                self.alloc_expr(Expr::OffsetOf(OffsetOf { container, fields }), syntax_ptr)
            }
            ast::Expr::FormatArgsExpr(f) => self.collect_format_args(f, syntax_ptr),
        })
    }

    fn collect_expr_path(&mut self, e: ast::PathExpr) -> Option<(Path, HygieneId)> {
        e.path().and_then(|path| {
            let path = self.lower_path(path, &mut Self::impl_trait_error_allocator)?;
            // Need to enable `mod_path.len() < 1` for `self`.
            let may_be_variable = matches!(&path, Path::BarePath(mod_path) if mod_path.len() <= 1);
            let hygiene = if may_be_variable {
                self.hygiene_id_for(e.syntax().text_range())
            } else {
                HygieneId::ROOT
            };
            Some((path, hygiene))
        })
    }

    /// Whether this path should be lowered as destructuring assignment, or as a normal assignment.
    fn path_is_destructuring_assignment(&self, path: &ModPath) -> bool {
        // rustc has access to a full resolver here, including local variables and generic params, and it checks the following
        // criteria: a path not lowered as destructuring assignment if it can *fully resolve* to something that is *not*
        // a const, a unit struct or a variant.
        // We don't have access to a full resolver here. So we should do the same as rustc, but assuming that local variables
        // could be resolved to nothing (fortunately, there cannot be a local variable shadowing a unit struct/variant/const,
        // as that is an error). We don't need to consider const params as it's an error to refer to these in patterns.
        let (resolution, unresolved_idx, _) = self.def_map.resolve_path_locally(
            self.local_def_map,
            self.db,
            self.module,
            path,
            BuiltinShadowMode::Other,
        );
        match unresolved_idx {
            Some(_) => {
                // If `Some(_)`, path could be resolved to unit struct/variant/const with type information, i.e. an assoc type or const.
                // If `None`, path could be a local variable.
                resolution.take_types().is_some()
            }
            None => match resolution.take_values() {
                // We don't need to consider non-unit structs/variants, as those are not value types.
                Some(ModuleDefId::EnumVariantId(_))
                | Some(ModuleDefId::AdtId(_))
                | Some(ModuleDefId::ConstId(_)) => true,
                _ => false,
            },
        }
    }

    fn collect_expr_as_pat_opt(&mut self, expr: Option<ast::Expr>) -> PatId {
        match expr {
            Some(expr) => self.collect_expr_as_pat(expr),
            _ => self.missing_pat(),
        }
    }

    fn collect_expr_as_pat(&mut self, expr: ast::Expr) -> PatId {
        self.maybe_collect_expr_as_pat(&expr).unwrap_or_else(|| {
            let src = self.expander.in_file(AstPtr::new(&expr).wrap_left());
            let expr = self.collect_expr(expr);
            // Do not use `alloc_pat_from_expr()` here, it will override the entry in `expr_map`.
            let id = self.store.pats.alloc(Pat::Expr(expr));
            self.store.pat_map_back.insert(id, src);
            id
        })
    }

    fn maybe_collect_expr_as_pat(&mut self, expr: &ast::Expr) -> Option<PatId> {
        if !self.check_cfg(expr) {
            return None;
        }
        let syntax_ptr = AstPtr::new(expr);

        let result = match expr {
            ast::Expr::UnderscoreExpr(_) => self.alloc_pat_from_expr(Pat::Wild, syntax_ptr),
            ast::Expr::ParenExpr(e) => {
                // We special-case `(..)` for consistency with patterns.
                if let Some(ast::Expr::RangeExpr(range)) = e.expr()
                    && range.is_range_full()
                {
                    return Some(self.alloc_pat_from_expr(
                        Pat::Tuple { args: Box::default(), ellipsis: Some(0) },
                        syntax_ptr,
                    ));
                }
                return e.expr().and_then(|expr| self.maybe_collect_expr_as_pat(&expr));
            }
            ast::Expr::TupleExpr(e) => {
                let (ellipsis, args) = collect_tuple(self, e.fields());
                self.alloc_pat_from_expr(Pat::Tuple { args, ellipsis }, syntax_ptr)
            }
            ast::Expr::ArrayExpr(e) => {
                if e.semicolon_token().is_some() {
                    return None;
                }

                let mut elements = e.exprs();
                let prefix = elements
                    .by_ref()
                    .map_while(|elem| collect_possibly_rest(self, elem).left())
                    .collect();
                let suffix = elements.map(|elem| self.collect_expr_as_pat(elem)).collect();
                self.alloc_pat_from_expr(Pat::Slice { prefix, slice: None, suffix }, syntax_ptr)
            }
            ast::Expr::CallExpr(e) => {
                let path = collect_path(self, e.expr()?)?;
                let path = path
                    .path()
                    .and_then(|path| self.lower_path(path, &mut Self::impl_trait_error_allocator));
                let Some(path) = path else {
                    return Some(self.missing_pat());
                };
                let (ellipsis, args) = collect_tuple(self, e.arg_list()?.args());
                self.alloc_pat_from_expr(Pat::TupleStruct { path, args, ellipsis }, syntax_ptr)
            }
            ast::Expr::PathExpr(e) => {
                let (path, hygiene) = self.collect_expr_path(e.clone())?;
                let mod_path = path.mod_path().expect("should not lower to lang path");
                if self.path_is_destructuring_assignment(mod_path) {
                    let pat_id = self.alloc_pat_from_expr(Pat::Path(path), syntax_ptr);
                    if !hygiene.is_root() {
                        self.store.ident_hygiene.insert(pat_id.into(), hygiene);
                    }
                    pat_id
                } else {
                    return None;
                }
            }
            ast::Expr::MacroExpr(e) => {
                let e = e.macro_call()?;
                let macro_ptr = AstPtr::new(&e);
                let src = self.expander.in_file(AstPtr::new(expr));
                let id = self.collect_macro_call(e, macro_ptr, true, |this, expansion| {
                    this.collect_expr_as_pat_opt(expansion)
                });
                self.store.expr_map.insert(src, id.into());
                id
            }
            ast::Expr::RecordExpr(e) => {
                let path = e
                    .path()
                    .and_then(|path| self.lower_path(path, &mut Self::impl_trait_error_allocator));
                let Some(path) = path else {
                    return Some(self.missing_pat());
                };
                let record_field_list = e.record_expr_field_list()?;
                let ellipsis = record_field_list.dotdot_token().is_some();
                if let Some(spread) = record_field_list.spread() {
                    self.store.diagnostics.push(
                        ExpressionStoreDiagnostics::FruInDestructuringAssignment {
                            node: self.expander.in_file(AstPtr::new(&spread)),
                        },
                    );
                }
                let args = record_field_list
                    .fields()
                    .filter_map(|f| {
                        if !self.check_cfg(&f) {
                            return None;
                        }
                        let field_expr = f.expr()?;
                        let pat = self.collect_expr_as_pat(field_expr);
                        let name = f.field_name()?.as_name();
                        let src = self.expander.in_file(AstPtr::new(&f).wrap_left());
                        self.store.pat_field_map_back.insert(pat, src);
                        Some(RecordFieldPat { name, pat })
                    })
                    .collect();
                self.alloc_pat_from_expr(Pat::Record { path, args, ellipsis }, syntax_ptr)
            }
            _ => return None,
        };
        return Some(result);

        fn collect_path(this: &mut ExprCollector<'_>, expr: ast::Expr) -> Option<ast::PathExpr> {
            match expr {
                ast::Expr::PathExpr(e) => Some(e),
                ast::Expr::MacroExpr(mac) => {
                    let call = mac.macro_call()?;
                    {
                        let macro_ptr = AstPtr::new(&call);
                        this.collect_macro_call(call, macro_ptr, true, |this, expanded_path| {
                            collect_path(this, expanded_path?)
                        })
                    }
                }
                _ => None,
            }
        }

        fn collect_possibly_rest(
            this: &mut ExprCollector<'_>,
            expr: ast::Expr,
        ) -> Either<PatId, ()> {
            match &expr {
                ast::Expr::RangeExpr(e) if e.is_range_full() => Either::Right(()),
                ast::Expr::MacroExpr(mac) => match mac.macro_call() {
                    Some(call) => {
                        let macro_ptr = AstPtr::new(&call);
                        let pat = this.collect_macro_call(
                            call,
                            macro_ptr,
                            true,
                            |this, expanded_expr| match expanded_expr {
                                Some(expanded_pat) => collect_possibly_rest(this, expanded_pat),
                                None => Either::Left(this.missing_pat()),
                            },
                        );
                        if let Either::Left(pat) = pat {
                            let src = this.expander.in_file(AstPtr::new(&expr).wrap_left());
                            this.store.pat_map_back.insert(pat, src);
                        }
                        pat
                    }
                    None => {
                        let ptr = AstPtr::new(&expr);
                        Either::Left(this.alloc_pat_from_expr(Pat::Missing, ptr))
                    }
                },
                _ => Either::Left(this.collect_expr_as_pat(expr)),
            }
        }

        fn collect_tuple(
            this: &mut ExprCollector<'_>,
            fields: ast::AstChildren<ast::Expr>,
        ) -> (Option<u32>, Box<[la_arena::Idx<Pat>]>) {
            let mut ellipsis = None;
            let args = fields
                .enumerate()
                .filter_map(|(idx, elem)| {
                    match collect_possibly_rest(this, elem) {
                        Either::Left(pat) => Some(pat),
                        Either::Right(()) => {
                            if ellipsis.is_none() {
                                ellipsis = Some(idx as u32);
                            }
                            // FIXME: Report an error here otherwise.
                            None
                        }
                    }
                })
                .collect();
            (ellipsis, args)
        }
    }

    /// The callback should return two exprs: the first is the bindings owner, the second is the expr to return.
    fn with_binding_owner_and_return(
        &mut self,
        create_expr: impl FnOnce(&mut Self) -> (ExprId, ExprId),
    ) -> ExprId {
        let prev_unowned_bindings_len = self.unowned_bindings.len();
        let (bindings_owner, expr_to_return) = create_expr(self);
        for binding in self.unowned_bindings.drain(prev_unowned_bindings_len..) {
            self.store.binding_owners.insert(binding, bindings_owner);
        }
        expr_to_return
    }

    fn with_binding_owner(&mut self, create_expr: impl FnOnce(&mut Self) -> ExprId) -> ExprId {
        self.with_binding_owner_and_return(move |this| {
            let expr = create_expr(this);
            (expr, expr)
        })
    }

    /// Desugar `try { <stmts>; <expr> }` into `'<new_label>: { <stmts>; ::std::ops::Try::from_output(<expr>) }`,
    /// `try { <stmts>; }` into `'<new_label>: { <stmts>; ::std::ops::Try::from_output(()) }`
    /// and save the `<new_label>` to use it as a break target for desugaring of the `?` operator.
    fn desugar_try_block(&mut self, e: BlockExpr, result_type: Option<ast::Type>) -> ExprId {
        let try_from_output = self.lang_path(self.lang_items().TryTraitFromOutput);
        let label = self.generate_new_name();
        let label = self.alloc_label_desugared(Label { name: label }, AstPtr::new(&e).wrap_right());
        let try_block_info = match result_type {
            Some(_) => TryBlock::Heterogeneous { label },
            None => TryBlock::Homogeneous { label },
        };
        let old_try_block = self.current_try_block.replace(try_block_info);

        let ptr = AstPtr::new(&e).upcast();
        let (btail, expr_id) = self.with_labeled_rib(label, HygieneId::ROOT, |this| {
            let mut btail = None;
            let block = this.collect_block_(e, |_, id, statements, tail| {
                btail = tail;
                Expr::Block { id, statements, tail, label: Some(label) }
            });
            (btail, block)
        });

        let callee = self
            .alloc_expr_desugared_with_ptr(try_from_output.map_or(Expr::Missing, Expr::Path), ptr);
        let next_tail = match btail {
            Some(tail) => self
                .alloc_expr_desugared_with_ptr(Expr::Call { callee, args: Box::new([tail]) }, ptr),
            None => {
                let unit =
                    self.alloc_expr_desugared_with_ptr(Expr::Tuple { exprs: Box::new([]) }, ptr);
                self.alloc_expr_desugared_with_ptr(
                    Expr::Call { callee, args: Box::new([unit]) },
                    ptr,
                )
            }
        };
        let Expr::Block { tail, .. } = &mut self.store.exprs[expr_id] else {
            unreachable!("block was lowered to non-block");
        };
        *tail = Some(next_tail);
        self.current_try_block = old_try_block;
        match result_type {
            Some(ty) => {
                // `{ let <name>: <ty> = <expr>; <name> }`
                let name = self.generate_new_name();
                let type_ref = self.lower_type_ref_disallow_impl_trait(ty);
                let binding = self.alloc_binding(
                    name.clone(),
                    BindingAnnotation::Unannotated,
                    HygieneId::ROOT,
                );
                let pat = self.alloc_pat_desugared(Pat::Bind { id: binding, subpat: None });
                self.add_definition_to_binding(binding, pat);
                let tail_expr =
                    self.alloc_expr_desugared_with_ptr(Expr::Path(Path::from(name)), ptr);
                self.alloc_expr_desugared_with_ptr(
                    Expr::Block {
                        id: None,
                        statements: Box::new([Statement::Let {
                            pat,
                            type_ref: Some(type_ref),
                            initializer: Some(expr_id),
                            else_branch: None,
                        }]),
                        tail: Some(tail_expr),
                        label: None,
                    },
                    ptr,
                )
            }
            None => expr_id,
        }
    }

    /// Desugar `ast::WhileExpr` from: `[opt_ident]: while <cond> <body>` into:
    /// ```ignore (pseudo-rust)
    /// [opt_ident]: loop {
    ///   if <cond> {
    ///     <body>
    ///   }
    ///   else {
    ///     break;
    ///   }
    /// }
    /// ```
    /// FIXME: Rustc wraps the condition in a construct equivalent to `{ let _t = <cond>; _t }`
    /// to preserve drop semantics. We should probably do the same in future.
    fn collect_while_loop(&mut self, syntax_ptr: AstPtr<ast::Expr>, e: ast::WhileExpr) -> ExprId {
        let label = e.label().map(|label| {
            (self.hygiene_id_for(label.syntax().text_range()), self.collect_label(label))
        });
        let body = self.collect_labelled_block_opt(label, e.loop_body());

        // Labels can also be used in the condition expression, like this:
        // ```
        // fn main() {
        //     let mut optional = Some(0);
        //     'my_label: while let Some(a) = match optional {
        //         None => break 'my_label,
        //         Some(val) => Some(val),
        //     } {
        //         println!("{}", a);
        //         optional = None;
        //     }
        // }
        // ```
        let condition = match label {
            Some((label_hygiene, label)) => self.with_labeled_rib(label, label_hygiene, |this| {
                this.collect_expr_opt(e.condition())
            }),
            None => self.collect_expr_opt(e.condition()),
        };

        let break_expr = self.alloc_expr(Expr::Break { expr: None, label: None }, syntax_ptr);
        let if_expr = self.alloc_expr(
            Expr::If { condition, then_branch: body, else_branch: Some(break_expr) },
            syntax_ptr,
        );
        self.alloc_expr(Expr::Loop { body: if_expr, label: label.map(|it| it.1) }, syntax_ptr)
    }

    /// Desugar `ast::ForExpr` from: `[opt_ident]: for <pat> in <head> <body>` into:
    /// ```ignore (pseudo-rust)
    /// match IntoIterator::into_iter(<head>) {
    ///     mut iter => {
    ///         [opt_ident]: loop {
    ///             match Iterator::next(&mut iter) {
    ///                 None => break,
    ///                 Some(<pat>) => <body>,
    ///             };
    ///         }
    ///     }
    /// }
    /// ```
    fn collect_for_loop(&mut self, syntax_ptr: AstPtr<ast::Expr>, e: ast::ForExpr) -> ExprId {
        let lang_items = self.lang_items();
        let (Some(into_iter_fn), Some(iter_next_fn), Some(option_some), Some(option_none)) = (
            self.lang_path(lang_items.IntoIterIntoIter),
            self.lang_path(lang_items.IteratorNext),
            self.lang_path(lang_items.OptionSome),
            self.lang_path(lang_items.OptionNone),
        ) else {
            return self.missing_expr();
        };
        let head = self.collect_expr_opt(e.iterable());
        let into_iter_fn_expr = self.alloc_expr(Expr::Path(into_iter_fn), syntax_ptr);
        let iterator = self.alloc_expr(
            Expr::Call { callee: into_iter_fn_expr, args: Box::new([head]) },
            syntax_ptr,
        );
        let none_arm = MatchArm {
            pat: self.alloc_pat_desugared(Pat::Path(option_none)),
            guard: None,
            expr: self.alloc_expr(Expr::Break { expr: None, label: None }, syntax_ptr),
        };
        let some_pat = Pat::TupleStruct {
            path: option_some,
            args: Box::new([self.collect_pat_top(e.pat())]),
            ellipsis: None,
        };
        let label = e.label().map(|label| {
            (self.hygiene_id_for(label.syntax().text_range()), self.collect_label(label))
        });
        let some_arm = MatchArm {
            pat: self.alloc_pat_desugared(some_pat),
            guard: None,
            expr: self.with_opt_labeled_rib(label, |this| {
                this.collect_expr_opt(e.loop_body().map(|it| it.into()))
            }),
        };
        let iter_name = self.generate_new_name();
        let iter_expr = self.alloc_expr(Expr::Path(Path::from(iter_name.clone())), syntax_ptr);
        let iter_expr_mut = self.alloc_expr(
            Expr::Ref { expr: iter_expr, rawness: Rawness::Ref, mutability: Mutability::Mut },
            syntax_ptr,
        );
        let iter_next_fn_expr = self.alloc_expr(Expr::Path(iter_next_fn), syntax_ptr);
        let iter_next_expr = self.alloc_expr(
            Expr::Call { callee: iter_next_fn_expr, args: Box::new([iter_expr_mut]) },
            syntax_ptr,
        );
        let loop_inner = self.alloc_expr(
            Expr::Match { expr: iter_next_expr, arms: Box::new([none_arm, some_arm]) },
            syntax_ptr,
        );
        let loop_inner = self.alloc_expr(
            Expr::Block {
                id: None,
                statements: Box::default(),
                tail: Some(loop_inner),
                label: None,
            },
            syntax_ptr,
        );
        let loop_outer = self
            .alloc_expr(Expr::Loop { body: loop_inner, label: label.map(|it| it.1) }, syntax_ptr);
        let iter_binding =
            self.alloc_binding(iter_name, BindingAnnotation::Mutable, HygieneId::ROOT);
        let iter_pat = self.alloc_pat_desugared(Pat::Bind { id: iter_binding, subpat: None });
        self.add_definition_to_binding(iter_binding, iter_pat);
        self.alloc_expr(
            Expr::Match {
                expr: iterator,
                arms: Box::new([MatchArm { pat: iter_pat, guard: None, expr: loop_outer }]),
            },
            syntax_ptr,
        )
    }

    /// Desugar `ast::TryExpr` from: `<expr>?` into:
    /// ```ignore (pseudo-rust)
    /// match Try::branch(<expr>) {
    ///     ControlFlow::Continue(val) => val,
    ///     ControlFlow::Break(residual) =>
    ///         // If there is an enclosing `try {...}`:
    ///         break 'catch_target Residual::into_try_type(residual),
    ///         // If there is an enclosing `try bikeshed Ty {...}`:
    ///         break 'catch_target Try::from_residual(residual),
    ///         // Otherwise:
    ///         return Try::from_residual(residual),
    /// }
    /// ```
    fn collect_try_operator(&mut self, syntax_ptr: AstPtr<ast::Expr>, e: ast::TryExpr) -> ExprId {
        let lang_items = self.lang_items();
        let (Some(try_branch), Some(cf_continue), Some(cf_break)) = (
            self.lang_path(lang_items.TryTraitBranch),
            self.lang_path(lang_items.ControlFlowContinue),
            self.lang_path(lang_items.ControlFlowBreak),
        ) else {
            return self.missing_expr();
        };
        let operand = self.collect_expr_opt(e.expr());
        let try_branch = self.alloc_expr(Expr::Path(try_branch), syntax_ptr);
        let expr = self
            .alloc_expr(Expr::Call { callee: try_branch, args: Box::new([operand]) }, syntax_ptr);
        let continue_name = self.generate_new_name();
        let continue_binding = self.alloc_binding(
            continue_name.clone(),
            BindingAnnotation::Unannotated,
            HygieneId::ROOT,
        );
        let continue_bpat =
            self.alloc_pat_desugared(Pat::Bind { id: continue_binding, subpat: None });
        self.add_definition_to_binding(continue_binding, continue_bpat);
        let continue_arm = MatchArm {
            pat: self.alloc_pat_desugared(Pat::TupleStruct {
                path: cf_continue,
                args: Box::new([continue_bpat]),
                ellipsis: None,
            }),
            guard: None,
            expr: self.alloc_expr(Expr::Path(Path::from(continue_name)), syntax_ptr),
        };
        let break_name = self.generate_new_name();
        let break_binding =
            self.alloc_binding(break_name.clone(), BindingAnnotation::Unannotated, HygieneId::ROOT);
        let break_bpat = self.alloc_pat_desugared(Pat::Bind { id: break_binding, subpat: None });
        self.add_definition_to_binding(break_binding, break_bpat);
        let break_arm = MatchArm {
            pat: self.alloc_pat_desugared(Pat::TupleStruct {
                path: cf_break,
                args: Box::new([break_bpat]),
                ellipsis: None,
            }),
            guard: None,
            expr: {
                let it = self.alloc_expr(Expr::Path(Path::from(break_name)), syntax_ptr);
                let convert_fn = match self.current_try_block {
                    Some(TryBlock::Homogeneous { .. }) => {
                        self.lang_path(lang_items.ResidualIntoTryType)
                    }
                    Some(TryBlock::Heterogeneous { .. }) | None => {
                        self.lang_path(lang_items.TryTraitFromResidual)
                    }
                };
                let callee =
                    self.alloc_expr(convert_fn.map_or(Expr::Missing, Expr::Path), syntax_ptr);
                let result =
                    self.alloc_expr(Expr::Call { callee, args: Box::new([it]) }, syntax_ptr);
                self.alloc_expr(
                    match self.current_try_block {
                        Some(
                            TryBlock::Heterogeneous { label } | TryBlock::Homogeneous { label },
                        ) => Expr::Break { expr: Some(result), label: Some(label) },
                        None => Expr::Return { expr: Some(result) },
                    },
                    syntax_ptr,
                )
            },
        };
        let arms = Box::new([continue_arm, break_arm]);
        self.alloc_expr(Expr::Match { expr, arms }, syntax_ptr)
    }

    fn collect_macro_call<T, U>(
        &mut self,
        mcall: ast::MacroCall,
        syntax_ptr: AstPtr<ast::MacroCall>,
        record_diagnostics: bool,
        collector: impl FnOnce(&mut Self, Option<T>) -> U,
    ) -> U
    where
        T: ast::AstNode,
    {
        let macro_call_ptr = self.expander.in_file(syntax_ptr);

        let block_call = self.def_map.modules[self.module].scope.macro_invoc(
            self.expander.in_file(self.expander.ast_id_map().ast_id_for_ptr(syntax_ptr)),
        );
        let res = match block_call {
            // fast path, macro call is in a block module
            Some(call) => Ok(self.expander.enter_expand_id(self.db, call)),
            None => {
                let resolver = |path: &_| {
                    self.def_map
                        .resolve_path(
                            self.local_def_map,
                            self.db,
                            self.module,
                            path,
                            crate::item_scope::BuiltinShadowMode::Other,
                            Some(MacroSubNs::Bang),
                        )
                        .0
                        .take_macros()
                };
                self.expander.enter_expand(
                    self.db,
                    mcall,
                    self.krate,
                    resolver,
                    &mut |ptr, call| {
                        _ = self.store.expansions.insert(ptr.map(|(it, _)| it), call);
                    },
                )
            }
        };

        let res = match res {
            Ok(res) => res,
            Err(UnresolvedMacro { path }) => {
                if record_diagnostics {
                    self.store.diagnostics.push(ExpressionStoreDiagnostics::UnresolvedMacroCall {
                        node: self.expander.in_file(syntax_ptr),
                        path,
                    });
                }
                return collector(self, None);
            }
        };
        // No need to push macro and parsing errors as they'll be recreated from `macro_calls()`.

        match res.value {
            Some((mark, expansion)) => {
                // Keep collecting even with expansion errors so we can provide completions and
                // other services in incomplete macro expressions.
                if let Some(macro_file) = self.expander.current_file_id().macro_file() {
                    self.store.expansions.insert(macro_call_ptr, macro_file);
                }

                let id = collector(self, expansion.map(|it| it.tree()));
                self.expander.exit(mark);
                id
            }
            None => collector(self, None),
        }
    }

    fn collect_macro_as_stmt(
        &mut self,
        statements: &mut Vec<Statement>,
        mac: ast::MacroExpr,
    ) -> Option<ExprId> {
        let mac_call = mac.macro_call()?;
        let syntax_ptr = AstPtr::new(&ast::Expr::from(mac));
        let macro_ptr = AstPtr::new(&mac_call);
        let expansion = self.collect_macro_call(
            mac_call,
            macro_ptr,
            false,
            |this, expansion: Option<ast::MacroStmts>| match expansion {
                Some(expansion) => {
                    expansion.statements().for_each(|stmt| this.collect_stmt(statements, stmt));
                    expansion.expr().and_then(|expr| match expr {
                        ast::Expr::MacroExpr(mac) => this.collect_macro_as_stmt(statements, mac),
                        expr => Some(this.collect_expr(expr)),
                    })
                }
                None => None,
            },
        );
        expansion.inspect(|&tail| {
            // Make the macro-call point to its expanded expression so we can query
            // semantics on syntax pointers to the macro
            let src = self.expander.in_file(syntax_ptr);
            self.store.expr_map.insert(src, tail.into());
        })
    }

    fn collect_stmt(&mut self, statements: &mut Vec<Statement>, s: ast::Stmt) {
        match s {
            ast::Stmt::LetStmt(stmt) => {
                if !self.check_cfg(&stmt) {
                    return;
                }
                let pat = self.collect_pat_top(stmt.pat());
                let type_ref = stmt.ty().map(|it| self.lower_type_ref_disallow_impl_trait(it));
                let initializer = stmt.initializer().map(|e| self.collect_expr(e));
                let else_branch = stmt
                    .let_else()
                    .and_then(|let_else| let_else.block_expr())
                    .map(|block| self.collect_block(block));
                statements.push(Statement::Let { pat, type_ref, initializer, else_branch });
            }
            ast::Stmt::ExprStmt(stmt) => {
                let expr = stmt.expr();
                match &expr {
                    Some(expr) if !self.check_cfg(expr) => return,
                    _ => (),
                }
                let has_semi = stmt.semicolon_token().is_some();
                // Note that macro could be expanded to multiple statements
                if let Some(ast::Expr::MacroExpr(mac)) = expr {
                    if let Some(expr) = self.collect_macro_as_stmt(statements, mac) {
                        statements.push(Statement::Expr { expr, has_semi })
                    }
                } else {
                    let expr = self.collect_expr_opt(expr);
                    statements.push(Statement::Expr { expr, has_semi });
                }
            }
            ast::Stmt::Item(ast::Item::MacroDef(macro_)) => {
                if !self.check_cfg(&macro_) {
                    return;
                }
                let Some(name) = macro_.name() else {
                    statements.push(Statement::Item(Item::Other));
                    return;
                };
                let name = name.as_name();
                let macro_id =
                    self.def_map.modules[self.def_map.root].scope.get(&name).take_macros();
                self.collect_macro_def(statements, macro_id);
            }
            ast::Stmt::Item(ast::Item::MacroRules(macro_)) => {
                if !self.check_cfg(&macro_) {
                    return;
                }
                let Some(name) = macro_.name() else {
                    statements.push(Statement::Item(Item::Other));
                    return;
                };
                let name = name.as_name();
                let macro_defs_count =
                    self.current_block_legacy_macro_defs_count.entry(name.clone()).or_insert(0);
                let macro_id = self.def_map.modules[self.def_map.root]
                    .scope
                    .get_legacy_macro(&name)
                    .and_then(|it| it.get(*macro_defs_count))
                    .copied();
                *macro_defs_count += 1;
                self.collect_macro_def(statements, macro_id);
            }
            ast::Stmt::Item(_item) => statements.push(Statement::Item(Item::Other)),
        }
    }

    fn collect_macro_def(&mut self, statements: &mut Vec<Statement>, macro_id: Option<MacroId>) {
        let Some(macro_id) = macro_id else {
            never!("def map should have macro definition, but it doesn't");
            statements.push(Statement::Item(Item::Other));
            return;
        };
        let macro_id = self.db.macro_def(macro_id);
        statements.push(Statement::Item(Item::MacroDef(Box::new(macro_id))));
        self.label_ribs.push(LabelRib::new(RibKind::MacroDef(Box::new(macro_id))));
    }

    fn collect_block(&mut self, block: ast::BlockExpr) -> ExprId {
        self.collect_block_(block, |_, id, statements, tail| Expr::Block {
            id,
            statements,
            tail,
            label: None,
        })
    }

    fn collect_block_(
        &mut self,
        block: ast::BlockExpr,
        mk_block: impl FnOnce(&mut Self, Option<BlockId>, Box<[Statement]>, Option<ExprId>) -> Expr,
    ) -> ExprId {
        let block_id = self.expander.ast_id_map().ast_id_for_block(&block).map(|file_local_id| {
            let ast_id = self.expander.in_file(file_local_id);
            BlockId::new(self.db, BlockLoc { ast_id, module: self.module })
        });

        let (module, def_map) =
            match block_id.map(|block_id| (block_def_map(self.db, block_id), block_id)) {
                Some((def_map, block_id)) => {
                    self.store.block_scopes.push(block_id);
                    (def_map.root_module_id(), def_map)
                }
                None => (self.module, self.def_map),
            };
        let prev_def_map = mem::replace(&mut self.def_map, def_map);
        let prev_local_module = mem::replace(&mut self.module, module);
        let prev_legacy_macros_count = mem::take(&mut self.current_block_legacy_macro_defs_count);

        let mut statements = Vec::new();
        block.statements().for_each(|s| self.collect_stmt(&mut statements, s));
        let tail = block.tail_expr().and_then(|e| match e {
            ast::Expr::MacroExpr(mac) => self.collect_macro_as_stmt(&mut statements, mac),
            expr => self.maybe_collect_expr(expr),
        });
        let tail = tail.or_else(|| {
            let stmt = statements.pop()?;
            if let Statement::Expr { expr, has_semi: false } = stmt {
                return Some(expr);
            }
            statements.push(stmt);
            None
        });

        let syntax_node_ptr = AstPtr::new(&block.into());
        let expr = mk_block(self, block_id, statements.into_boxed_slice(), tail);
        let expr_id = self.alloc_expr(expr, syntax_node_ptr);

        self.def_map = prev_def_map;
        self.module = prev_local_module;
        self.current_block_legacy_macro_defs_count = prev_legacy_macros_count;
        expr_id
    }

    fn collect_block_opt(&mut self, expr: Option<ast::BlockExpr>) -> ExprId {
        match expr {
            Some(block) => self.collect_block(block),
            None => self.missing_expr(),
        }
    }

    fn collect_labelled_block_opt(
        &mut self,
        label: Option<(HygieneId, LabelId)>,
        expr: Option<ast::BlockExpr>,
    ) -> ExprId {
        match label {
            Some((hygiene, label)) => {
                self.with_labeled_rib(label, hygiene, |this| this.collect_block_opt(expr))
            }
            None => self.collect_block_opt(expr),
        }
    }

    fn collect_extern_fn_param(&mut self, pat: Option<ast::Pat>) -> PatId {
        // `extern` functions cannot have pattern-matched parameters, and furthermore, the identifiers
        // in their parameters are always interpreted as bindings, even if in a normal function they
        // won't be, because they would refer to a path pattern.
        let Some(pat) = pat else { return self.missing_pat() };

        match &pat {
            ast::Pat::IdentPat(bp) if bp.is_simple_ident() => {
                let name = bp.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                let hygiene = bp
                    .name()
                    .map(|name| self.hygiene_id_for(name.syntax().text_range()))
                    .unwrap_or(HygieneId::ROOT);
                let binding = self.alloc_binding(name, BindingAnnotation::Unannotated, hygiene);
                let pat =
                    self.alloc_pat(Pat::Bind { id: binding, subpat: None }, AstPtr::new(&pat));
                self.add_definition_to_binding(binding, pat);
                pat
            }
            _ => {
                self.store.diagnostics.push(ExpressionStoreDiagnostics::PatternArgInExternFn {
                    node: self.expander.in_file(AstPtr::new(&pat)),
                });
                self.missing_pat()
            }
        }
    }

    // region: patterns

    fn collect_pat_top(&mut self, pat: Option<ast::Pat>) -> PatId {
        match pat {
            Some(pat) => self.collect_pat(pat, &mut BindingList::default()),
            None => self.missing_pat(),
        }
    }

    fn collect_pat(&mut self, pat: ast::Pat, binding_list: &mut BindingList) -> PatId {
        let pattern = match &pat {
            ast::Pat::IdentPat(bp) => {
                let name = bp.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                let hygiene = bp
                    .name()
                    .map(|name| self.hygiene_id_for(name.syntax().text_range()))
                    .unwrap_or(HygieneId::ROOT);

                let annotation =
                    BindingAnnotation::new(bp.mut_token().is_some(), bp.ref_token().is_some());
                let subpat = bp.pat().map(|subpat| self.collect_pat(subpat, binding_list));

                let is_simple_ident_pat =
                    annotation == BindingAnnotation::Unannotated && subpat.is_none();
                let (binding, pattern) = if is_simple_ident_pat {
                    // This could also be a single-segment path pattern. To
                    // decide that, we need to try resolving the name.
                    let (resolved, _) = self.def_map.resolve_path(
                        self.local_def_map,
                        self.db,
                        self.module,
                        &name.clone().into(),
                        BuiltinShadowMode::Other,
                        None,
                    );
                    // Funnily enough, record structs/variants *can* be shadowed
                    // by pattern bindings (but unit or tuple structs/variants
                    // can't).
                    match resolved.take_values() {
                        Some(ModuleDefId::ConstId(_)) => (None, Pat::Path(name.into())),
                        Some(ModuleDefId::EnumVariantId(variant))
                        // FIXME: This can cause a cycle if the user is writing invalid code
                            if variant.fields(self.db).shape != FieldsShape::Record =>
                        {
                            (None, Pat::Path(name.into()))
                        }
                        Some(ModuleDefId::AdtId(AdtId::StructId(s)))
                        // FIXME: This can cause a cycle if the user is writing invalid code
                            if StructSignature::of(self.db, s).shape != FieldsShape::Record =>
                        {
                            (None, Pat::Path(name.into()))
                        }
                        // shadowing statics is an error as well, so we just ignore that case here
                        _ => {
                            let id = binding_list.find(self, name, hygiene, annotation);
                            (Some(id), Pat::Bind { id, subpat })
                        }
                    }
                } else {
                    let id = binding_list.find(self, name, hygiene, annotation);
                    (Some(id), Pat::Bind { id, subpat })
                };

                let ptr = AstPtr::new(&pat);
                let pat = self.alloc_pat(pattern, ptr);
                if let Some(binding_id) = binding {
                    self.add_definition_to_binding(binding_id, pat);
                }
                return pat;
            }
            ast::Pat::TupleStructPat(p) => {
                let path = p
                    .path()
                    .and_then(|path| self.lower_path(path, &mut Self::impl_trait_error_allocator));
                let Some(path) = path else {
                    return self.missing_pat();
                };
                let (args, ellipsis) = self.collect_tuple_pat(
                    p.fields(),
                    comma_follows_token(p.l_paren_token()),
                    binding_list,
                );
                Pat::TupleStruct { path, args, ellipsis }
            }
            ast::Pat::RefPat(p) => {
                let pat = self.collect_pat_opt(p.pat(), binding_list);
                let mutability = Mutability::from_mutable(p.mut_token().is_some());
                Pat::Ref { pat, mutability }
            }
            ast::Pat::PathPat(p) => {
                let path = p
                    .path()
                    .and_then(|path| self.lower_path(path, &mut Self::impl_trait_error_allocator));
                path.map(Pat::Path).unwrap_or(Pat::Missing)
            }
            ast::Pat::OrPat(p) => 'b: {
                let prev_is_used = mem::take(&mut binding_list.is_used);
                let prev_reject_new = mem::take(&mut binding_list.reject_new);
                let mut pats = Vec::with_capacity(p.pats().count());
                let mut it = p.pats();
                let Some(first) = it.next() else {
                    break 'b Pat::Or(Box::new([]));
                };
                pats.push(self.collect_pat(first, binding_list));
                binding_list.reject_new = true;
                for rest in it {
                    for (_, it) in binding_list.is_used.iter_mut() {
                        *it = false;
                    }
                    pats.push(self.collect_pat(rest, binding_list));
                    for (&id, &is_used) in binding_list.is_used.iter() {
                        if !is_used {
                            self.store.bindings[id].problems =
                                Some(BindingProblems::NotBoundAcrossAll);
                        }
                    }
                }
                binding_list.reject_new = prev_reject_new;
                let current_is_used = mem::replace(&mut binding_list.is_used, prev_is_used);
                for (id, _) in current_is_used.into_iter() {
                    binding_list.check_is_used(self, id);
                }
                if let &[pat] = &*pats {
                    // Leading pipe without real OR pattern. Leaving an one-item OR pattern may confuse later stages.
                    return pat;
                }
                Pat::Or(pats.into())
            }
            ast::Pat::ParenPat(p) => return self.collect_pat_opt(p.pat(), binding_list),
            ast::Pat::TuplePat(p) => {
                let (args, ellipsis) = self.collect_tuple_pat(
                    p.fields(),
                    comma_follows_token(p.l_paren_token()),
                    binding_list,
                );
                Pat::Tuple { args, ellipsis }
            }
            ast::Pat::WildcardPat(_) => Pat::Wild,
            ast::Pat::RecordPat(p) => {
                let path = p
                    .path()
                    .and_then(|path| self.lower_path(path, &mut Self::impl_trait_error_allocator));
                let Some(path) = path else {
                    return self.missing_pat();
                };
                let record_pat_field_list =
                    &p.record_pat_field_list().expect("every struct should have a field list");
                let args = record_pat_field_list
                    .fields()
                    .filter_map(|f| {
                        if !self.check_cfg(&f) {
                            return None;
                        }
                        let ast_pat = f.pat()?;
                        let pat = self.collect_pat(ast_pat, binding_list);
                        let name = f.field_name()?.as_name();
                        let src = self.expander.in_file(AstPtr::new(&f).wrap_right());
                        self.store.pat_field_map_back.insert(pat, src);
                        Some(RecordFieldPat { name, pat })
                    })
                    .collect();

                let ellipsis = record_pat_field_list.rest_pat().is_some();

                Pat::Record { path, args, ellipsis }
            }
            ast::Pat::SlicePat(p) => {
                let SlicePatComponents { prefix, slice, suffix } = p.components();

                Pat::Slice {
                    prefix: prefix.into_iter().map(|p| self.collect_pat(p, binding_list)).collect(),
                    slice: slice.map(|p| self.collect_pat(p, binding_list)),
                    suffix: suffix.into_iter().map(|p| self.collect_pat(p, binding_list)).collect(),
                }
            }
            ast::Pat::LiteralPat(lit) => 'b: {
                let Some((hir_lit, ast_lit)) = pat_literal_to_hir(lit) else {
                    break 'b Pat::Missing;
                };
                let expr = Expr::Literal(hir_lit);
                let expr_ptr = AstPtr::new(&ast::Expr::Literal(ast_lit));
                let expr_id = self.alloc_expr(expr, expr_ptr);
                Pat::Lit(expr_id)
            }
            ast::Pat::RestPat(_) => Pat::Rest,
            ast::Pat::BoxPat(boxpat) => {
                let inner = self.collect_pat_opt(boxpat.pat(), binding_list);
                Pat::Box { inner }
            }
            ast::Pat::DerefPat(inner) => {
                let inner = self.collect_pat_opt(inner.pat(), binding_list);
                Pat::Deref { inner }
            }
            ast::Pat::NotNull(_) => Pat::NotNull,
            ast::Pat::ConstBlockPat(const_block_pat) => {
                if let Some(block) = const_block_pat.block_expr() {
                    let expr_id = self.with_label_rib(RibKind::Constant, |this| {
                        this.with_binding_owner(|this| this.collect_block(block))
                    });
                    Pat::ConstBlock(expr_id)
                } else {
                    Pat::Missing
                }
            }
            ast::Pat::MacroPat(mac) => match mac.macro_call() {
                Some(call) => {
                    let macro_ptr = AstPtr::new(&call);
                    let src = self.expander.in_file(AstPtr::new(&pat));
                    let pat =
                        self.collect_macro_call(call, macro_ptr, true, |this, expanded_pat| {
                            this.collect_pat_opt(expanded_pat, binding_list)
                        });
                    self.store.pat_map.insert(src, pat.into());
                    return pat;
                }
                None => Pat::Missing,
            },
            ast::Pat::RangePat(p) => {
                let mut range_part_lower = |p: Option<ast::Pat>| -> Option<ExprId> {
                    p.and_then(|it| {
                        let ptr = PatPtr::new(&it);
                        match &it {
                            ast::Pat::LiteralPat(it) => Some(self.alloc_expr_from_pat(
                                Expr::Literal(pat_literal_to_hir(it)?.0),
                                ptr,
                            )),
                            ast::Pat::IdentPat(ident) if ident.is_simple_ident() => ident
                                .name()
                                .map(|name| name.as_name())
                                .map(Path::from)
                                .map(|path| self.alloc_expr_from_pat(Expr::Path(path), ptr)),
                            ast::Pat::PathPat(p) => p
                                .path()
                                .and_then(|path| {
                                    self.lower_path(path, &mut Self::impl_trait_error_allocator)
                                })
                                .map(|parsed| self.alloc_expr_from_pat(Expr::Path(parsed), ptr)),
                            // We only need to handle literal, ident (if bare) and path patterns here,
                            // as any other pattern as a range pattern operand is semantically invalid.
                            _ => None,
                        }
                    })
                };
                let start = range_part_lower(p.start());
                let end = range_part_lower(p.end());
                match p.op_kind() {
                    Some(range_type) => Pat::Range { start, end, range_type },
                    None => Pat::Missing,
                }
            }
        };
        let ptr = AstPtr::new(&pat);
        self.alloc_pat(pattern, ptr)
    }

    fn collect_pat_opt(&mut self, pat: Option<ast::Pat>, binding_list: &mut BindingList) -> PatId {
        match pat {
            Some(pat) => self.collect_pat(pat, binding_list),
            None => self.missing_pat(),
        }
    }

    fn collect_tuple_pat(
        &mut self,
        args: AstChildren<ast::Pat>,
        has_leading_comma: bool,
        binding_list: &mut BindingList,
    ) -> (Box<[PatId]>, Option<u32>) {
        let args: Vec<_> = args.map(|p| self.collect_pat_possibly_rest(p, binding_list)).collect();
        // Find the location of the `..`, if there is one. Note that we do not
        // consider the possibility of there being multiple `..` here.
        let ellipsis = args.iter().position(|p| p.is_right()).map(|it| it as u32);

        // We want to skip the `..` pattern here, since we account for it above.
        let mut args: Vec<_> = args.into_iter().filter_map(Either::left).collect();
        // if there is a leading comma, the user is most likely to type out a leading pattern
        // so we insert a missing pattern at the beginning for IDE features
        if has_leading_comma {
            args.insert(0, self.missing_pat());
        }

        (args.into_boxed_slice(), ellipsis)
    }

    // `collect_pat` rejects `ast::Pat::RestPat`, but it should be handled in some cases that
    // it is the macro expansion result of an arg sub-pattern in a slice or tuple pattern.
    fn collect_pat_possibly_rest(
        &mut self,
        pat: ast::Pat,
        binding_list: &mut BindingList,
    ) -> Either<PatId, ()> {
        match &pat {
            ast::Pat::RestPat(_) => Either::Right(()),
            ast::Pat::MacroPat(mac) => match mac.macro_call() {
                Some(call) => {
                    let macro_ptr = AstPtr::new(&call);
                    let src = self.expander.in_file(AstPtr::new(&pat));
                    let pat =
                        self.collect_macro_call(call, macro_ptr, true, |this, expanded_pat| {
                            if let Some(expanded_pat) = expanded_pat {
                                this.collect_pat_possibly_rest(expanded_pat, binding_list)
                            } else {
                                Either::Left(this.missing_pat())
                            }
                        });
                    if let Some(pat) = pat.left() {
                        self.store.pat_map.insert(src, pat.into());
                    }
                    pat
                }
                None => {
                    let ptr = AstPtr::new(&pat);
                    Either::Left(self.alloc_pat(Pat::Missing, ptr))
                }
            },
            _ => Either::Left(self.collect_pat(pat, binding_list)),
        }
    }

    fn collect_ty_pat_opt(&mut self, pat: Option<ast::Pat>) -> PatId {
        match pat {
            Some(pat) => self.collect_ty_pat(pat),
            None => self.missing_pat(),
        }
    }

    fn collect_ty_pat(&mut self, pat: ast::Pat) -> PatId {
        let ptr = AstPtr::new(&pat);
        match pat {
            ast::Pat::NotNull(_) => self.alloc_pat(Pat::NotNull, ptr),
            ast::Pat::OrPat(pat) => {
                let pat = pat.pats().map(|pat| self.collect_ty_pat(pat)).collect();
                self.alloc_pat(Pat::Or(pat), ptr)
            }
            ast::Pat::RangePat(range_pat) => {
                let start = range_pat
                    .start()
                    .map(|pat| {
                        self.with_fresh_binding_expr_root(|this| this.lower_ty_pat_range_side(pat))
                    })
                    .unwrap_or_else(|| self.lower_ty_pat_range_end(self.lang_items().RangeMin));
                let end = range_pat
                    .end()
                    .map(|pat| match range_pat.op_kind() {
                        Some(ast::RangeOp::Inclusive) | None => self
                            .with_fresh_binding_expr_root(|this| this.lower_ty_pat_range_side(pat)),
                        Some(ast::RangeOp::Exclusive) => self.lower_excluded_range_end(pat),
                    })
                    .unwrap_or_else(|| self.lower_ty_pat_range_end(self.lang_items().RangeMax));
                self.alloc_pat(
                    Pat::Range {
                        start: Some(start),
                        end: Some(end),
                        range_type: ast::RangeOp::Inclusive,
                    },
                    ptr,
                )
            }
            ast::Pat::MacroPat(pat) => {
                let Some(call) = pat.macro_call() else { return self.missing_pat() };
                let ptr = AstPtr::new(&call);
                self.collect_macro_call(call, ptr, true, |this, pat| this.collect_ty_pat_opt(pat))
            }
            _ => {
                // FIXME: Emit an error.
                self.alloc_pat(Pat::Missing, ptr)
            }
        }
    }

    fn lower_ty_pat_range_side(&mut self, pat: ast::Pat) -> ExprId {
        match &pat {
            ast::Pat::LiteralPat(it) => {
                let Some((literal, _)) = pat_literal_to_hir(it) else { return self.missing_expr() };
                self.alloc_expr_from_pat(Expr::Literal(literal), AstPtr::new(&pat))
            }
            _ => self.missing_expr(),
        }
    }

    /// When a range has no end specified (`1..` or `1..=`) or no start specified (`..5` or `..=5`),
    /// we instead use a constant of the MAX/MIN of the type.
    /// This way the type system does not have to handle the lack of a start/end.
    fn lower_ty_pat_range_end(&mut self, lang_item: Option<ConstId>) -> ExprId {
        self.with_fresh_binding_expr_root(|this| {
            this.alloc_expr_desugared(
                this.lang_path(lang_item).map(Expr::Path).unwrap_or(Expr::Missing),
            )
        })
    }

    /// Lowers the range end of an exclusive range (`2..5`) to an inclusive range 2..=(5 - 1).
    /// This way the type system doesn't have to handle the distinction between inclusive/exclusive ranges.
    fn lower_excluded_range_end(&mut self, pat: ast::Pat) -> ExprId {
        self.with_fresh_binding_expr_root(|this| {
            let excluded_end = this.lower_ty_pat_range_side(pat);
            let range_sub_path =
                this.lang_path(this.lang_items().RangeSub).map(Expr::Path).unwrap_or(Expr::Missing);
            let range_sub_path = this.alloc_expr_desugared(range_sub_path);
            this.alloc_expr_desugared(Expr::Call {
                callee: range_sub_path,
                args: Box::new([excluded_end]),
            })
        })
    }

    // endregion: patterns

    /// Returns `false` (and emits diagnostics) when `owner` if `#[cfg]`d out, and `true` when
    /// not.
    fn check_cfg(&mut self, owner: &dyn ast::HasAttrs) -> bool {
        let enabled = self.expander.is_cfg_enabled(owner, self.cfg_options);
        match enabled {
            Ok(()) => true,
            Err(cfg) => {
                self.store.diagnostics.push(ExpressionStoreDiagnostics::InactiveCode {
                    node: self.expander.in_file(SyntaxNodePtr::new(owner.syntax())),
                    cfg,
                    opts: self.cfg_options.clone(),
                });
                false
            }
        }
    }

    fn add_definition_to_binding(&mut self, binding_id: BindingId, pat_id: PatId) {
        self.store.binding_definitions.entry(binding_id).or_default().push(pat_id);
    }

    // region: labels

    fn collect_label(&mut self, ast_label: ast::Label) -> LabelId {
        let label = Label {
            name: ast_label
                .lifetime()
                .as_ref()
                .map_or_else(Name::missing, |lt| Name::new_lifetime(&lt.text())),
        };
        self.alloc_label(label, AstPtr::new(&ast_label))
    }

    fn resolve_label(
        &self,
        lifetime: Option<ast::Lifetime>,
    ) -> Result<Option<LabelId>, ExpressionStoreDiagnostics> {
        let Some(lifetime) = lifetime else { return Ok(None) };
        let mut hygiene_id =
            self.expander.hygiene_for_range(self.db, lifetime.syntax().text_range());
        let mut hygiene_info = if hygiene_id.is_root() {
            None
        } else {
            hygiene_id.syntax_context().outer_expn(self.db).map(|expansion| {
                let expansion = hir_expand::MacroCallId::from(expansion).loc(self.db);
                (hygiene_id.syntax_context().parent(self.db), expansion.def)
            })
        };
        let name = Name::new_lifetime(&lifetime.text());

        for (rib_idx, rib) in self.label_ribs.iter().enumerate().rev() {
            match &rib.kind {
                RibKind::Normal(label_name, id, label_hygiene)
                    if *label_name == name && *label_hygiene == hygiene_id =>
                {
                    return if self.is_label_valid_from_rib(rib_idx) {
                        Ok(Some(*id))
                    } else {
                        Err(ExpressionStoreDiagnostics::UnreachableLabel {
                            name,
                            node: self.expander.in_file(AstPtr::new(&lifetime)),
                        })
                    };
                }
                RibKind::MacroDef(macro_id) => {
                    if let Some((parent_ctx, label_macro_id)) = hygiene_info
                        && label_macro_id == **macro_id
                    {
                        // A macro is allowed to refer to labels from before its declaration.
                        // Therefore, if we got to the rib of its declaration, give up its hygiene
                        // and use its parent expansion.

                        hygiene_id = HygieneId::new(parent_ctx.opaque_and_semiopaque(self.db));
                        hygiene_info = parent_ctx.outer_expn(self.db).map(|expansion| {
                            let expansion = hir_expand::MacroCallId::from(expansion).loc(self.db);
                            (parent_ctx.parent(self.db), expansion.def)
                        });
                    }
                }
                _ => {}
            }
        }

        Err(ExpressionStoreDiagnostics::UndeclaredLabel {
            name,
            node: self.expander.in_file(AstPtr::new(&lifetime)),
        })
    }

    fn is_label_valid_from_rib(&self, rib_index: usize) -> bool {
        !self.label_ribs[rib_index + 1..].iter().any(|rib| rib.kind.is_label_barrier())
    }

    fn pop_label_rib(&mut self) {
        // We need to pop all macro defs, plus one rib.
        while let Some(LabelRib { kind: RibKind::MacroDef(_) }) = self.label_ribs.pop() {
            // Do nothing.
        }
    }

    fn with_label_rib<T>(&mut self, kind: RibKind, f: impl FnOnce(&mut Self) -> T) -> T {
        self.label_ribs.push(LabelRib::new(kind));
        let res = f(self);
        self.pop_label_rib();
        res
    }

    fn with_labeled_rib<T>(
        &mut self,
        label: LabelId,
        hygiene: HygieneId,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.label_ribs.push(LabelRib::new(RibKind::Normal(
            self.store.labels[label].name.clone(),
            label,
            hygiene,
        )));
        let res = f(self);
        self.pop_label_rib();
        res
    }

    fn with_opt_labeled_rib<T>(
        &mut self,
        label: Option<(HygieneId, LabelId)>,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        match label {
            None => f(self),
            Some((hygiene, label)) => self.with_labeled_rib(label, hygiene, f),
        }
    }
    // endregion: labels

    fn expand_macros_to_string(&mut self, expr: ast::Expr) -> Option<(ast::String, bool)> {
        let m = match expr {
            ast::Expr::MacroExpr(m) => m,
            ast::Expr::Literal(l) => {
                return match l.kind() {
                    ast::LiteralKind::String(s) => Some((s, true)),
                    _ => None,
                };
            }
            _ => return None,
        };
        let e = m.macro_call()?;
        let macro_ptr = AstPtr::new(&e);
        let (exp, _) = self.collect_macro_call(e, macro_ptr, true, |this, expansion| {
            expansion.and_then(|it| this.expand_macros_to_string(it))
        })?;
        Some((exp, false))
    }

    fn lang_path(&self, lang: Option<impl Into<LangItemTarget>>) -> Option<Path> {
        Some(Path::LangItem(lang?.into(), None))
    }

    fn ty_rel_lang_path(
        &self,
        lang: Option<impl Into<LangItemTarget>>,
        relative_name: Symbol,
    ) -> Option<Path> {
        Some(Path::LangItem(lang?.into(), Some(Name::new_symbol_root(relative_name))))
    }

    fn ty_rel_lang_path_expr(
        &self,
        lang: Option<impl Into<LangItemTarget>>,
        relative_name: Symbol,
    ) -> Expr {
        self.ty_rel_lang_path(lang, relative_name).map_or(Expr::Missing, Expr::Path)
    }
}

fn pat_literal_to_hir(lit: &ast::LiteralPat) -> Option<(Literal, ast::Literal)> {
    let ast_lit = lit.literal()?;
    let mut hir_lit: Literal = ast_lit.kind().into();
    if lit.minus_token().is_some() {
        hir_lit = hir_lit.negate()?;
    }
    Some((hir_lit, ast_lit))
}

impl ExprCollector<'_> {
    fn with_fresh_binding_expr_root(&mut self, f: impl FnOnce(&mut Self) -> ExprId) -> ExprId {
        self.with_expr_root(|this| this.with_binding_owner(f))
    }

    fn with_expr_root(&mut self, f: impl FnOnce(&mut Self) -> ExprId) -> ExprId {
        let inference_roots = self.store.inference_roots.take();
        let root = f(self);
        self.store.inference_roots = inference_roots;

        if let Some(inference_roots) = &mut self.store.inference_roots {
            inference_roots.push(ExprRoot {
                root,
                exprs_end: end(&self.store.exprs),
                pats_end: end(&self.store.pats),
                bindings_end: end(&self.store.bindings),
            });
        }

        return root;

        fn end<T>(arena: &la_arena::Arena<T>) -> la_arena::Idx<T> {
            la_arena::Idx::from_raw(la_arena::RawIdx::from_u32(arena.len() as u32))
        }
    }

    fn alloc_expr(&mut self, expr: Expr, ptr: ExprPtr) -> ExprId {
        let src = self.expander.in_file(ptr);
        let id = self.store.exprs.alloc(expr);
        self.store.expr_map_back.insert(id, src.map(AstPtr::wrap_left));
        self.store.expr_map.insert(src, id.into());
        id
    }
    // FIXME: desugared exprs don't have ptr, that's wrong and should be fixed.
    // Migrate to alloc_expr_desugared_with_ptr and then rename back
    fn alloc_expr_desugared(&mut self, expr: Expr) -> ExprId {
        self.store.exprs.alloc(expr)
    }
    fn alloc_expr_desugared_with_ptr(&mut self, expr: Expr, ptr: ExprPtr) -> ExprId {
        let src = self.expander.in_file(ptr);
        let id = self.store.exprs.alloc(expr);
        self.store.expr_map_back.insert(id, src.map(AstPtr::wrap_left));
        // We intentionally don't fill this as it could overwrite a non-desugared entry
        // self.store.expr_map.insert(src, id);
        id
    }
    fn missing_expr(&mut self) -> ExprId {
        self.alloc_expr_desugared(Expr::Missing)
    }

    fn alloc_binding(
        &mut self,
        name: Name,
        mode: BindingAnnotation,
        hygiene: HygieneId,
    ) -> BindingId {
        let binding = self.store.bindings.alloc(Binding { name, mode, problems: None, hygiene });
        self.unowned_bindings.push(binding);
        binding
    }

    fn alloc_pat_from_expr(&mut self, pat: Pat, ptr: ExprPtr) -> PatId {
        let src = self.expander.in_file(ptr);
        let id = self.store.pats.alloc(pat);
        self.store.expr_map.insert(src, id.into());
        self.store.pat_map_back.insert(id, src.map(AstPtr::wrap_left));
        id
    }

    fn alloc_expr_from_pat(&mut self, expr: Expr, ptr: PatPtr) -> ExprId {
        let src = self.expander.in_file(ptr);
        let id = self.store.exprs.alloc(expr);
        self.store.pat_map.insert(src, id.into());
        self.store.expr_map_back.insert(id, src.map(AstPtr::wrap_right));
        id
    }

    fn alloc_pat(&mut self, pat: Pat, ptr: PatPtr) -> PatId {
        let src = self.expander.in_file(ptr);
        let id = self.store.pats.alloc(pat);
        self.store.pat_map_back.insert(id, src.map(AstPtr::wrap_right));
        self.store.pat_map.insert(src, id.into());
        id
    }
    // FIXME: desugared pats don't have ptr, that's wrong and should be fixed somehow.
    fn alloc_pat_desugared(&mut self, pat: Pat) -> PatId {
        self.store.pats.alloc(pat)
    }
    fn missing_pat(&mut self) -> PatId {
        self.store.pats.alloc(Pat::Missing)
    }

    fn alloc_label(&mut self, label: Label, ptr: AstPtr<ast::Label>) -> LabelId {
        self.alloc_label_desugared(label, ptr.wrap_left())
    }

    fn alloc_label_desugared(&mut self, label: Label, ptr: LabelPtr) -> LabelId {
        let src = self.expander.in_file(ptr);
        let id = self.store.labels.alloc(label);
        self.store.label_map_back.insert(id, src);
        self.store.label_map.insert(src, id);
        id
    }

    fn is_lowering_awaitable_block(&self) -> &Awaitable {
        self.awaitable_context.as_ref().unwrap_or(&Awaitable::No("unknown"))
    }

    fn with_awaitable_block<T>(
        &mut self,
        awaitable: Awaitable,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let orig = self.awaitable_context.replace(awaitable);
        let res = f(self);
        self.awaitable_context = orig;
        res
    }

    fn hygiene_id_for(&self, range: TextRange) -> HygieneId {
        self.expander.hygiene_for_range(self.db, range)
    }
}

fn comma_follows_token(t: Option<syntax::SyntaxToken>) -> bool {
    (|| syntax::algo::skip_trivia_token(t?.next_token()?, syntax::Direction::Next))()
        .is_some_and(|it| it.kind() == syntax::T![,])
}

/// This function find the AST fragment that corresponds to an `AssociatedTypeBinding` in the HIR.
pub fn hir_assoc_type_binding_to_ast(
    segment_args: &ast::GenericArgList,
    binding_idx: u32,
) -> Option<ast::AssocTypeArg> {
    segment_args
        .generic_args()
        .filter_map(|arg| match arg {
            ast::GenericArg::AssocTypeArg(it) => Some(it),
            _ => None,
        })
        .filter(|binding| binding.param_list().is_none() && binding.name_ref().is_some())
        .nth(binding_idx as usize)
}

/// This function find the AST generic argument from the one in the HIR. Does not support the `Self` argument.
pub fn hir_generic_arg_to_ast(
    args: &ast::GenericArgList,
    arg_idx: u32,
    has_self_arg: bool,
) -> Option<ast::GenericArg> {
    args.generic_args()
        .filter(|arg| match arg {
            ast::GenericArg::AssocTypeArg(_) => false,
            ast::GenericArg::LifetimeArg(arg) => arg.lifetime().is_some(),
            ast::GenericArg::ConstArg(_) | ast::GenericArg::TypeArg(_) => true,
        })
        .nth(arg_idx as usize - has_self_arg as usize)
}
