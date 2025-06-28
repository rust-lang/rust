//! Transforms `ast::Expr` into an equivalent `hir_def::expr::Expr`
//! representation.

mod asm;
mod generics;
mod path;

use std::mem;

use base_db::FxIndexSet;
use cfg::CfgOptions;
use either::Either;
use hir_expand::{
    HirFileId, InFile, MacroDefId,
    mod_path::tool_path,
    name::{AsName, Name},
    span_map::SpanMapRef,
};
use intern::{Symbol, sym};
use rustc_hash::FxHashMap;
use stdx::never;
use syntax::{
    AstNode, AstPtr, AstToken as _, SyntaxNodePtr,
    ast::{
        self, ArrayExprKind, AstChildren, BlockExpr, HasArgList, HasAttrs, HasGenericArgs,
        HasGenericParams, HasLoopBody, HasName, HasTypeBounds, IsString, RangeItem,
        SlicePatComponents,
    },
};
use thin_vec::ThinVec;
use triomphe::Arc;
use tt::TextRange;

use crate::{
    AdtId, BlockId, BlockLoc, DefWithBodyId, FunctionId, GenericDefId, ImplId, MacroId,
    ModuleDefId, ModuleId, TraitAliasId, TraitId, TypeAliasId, UnresolvedMacro,
    builtin_type::BuiltinUint,
    db::DefDatabase,
    expr_store::{
        Body, BodySourceMap, ExprPtr, ExpressionStore, ExpressionStoreBuilder,
        ExpressionStoreDiagnostics, ExpressionStoreSourceMap, HygieneId, LabelPtr, LifetimePtr,
        PatPtr, TypePtr,
        expander::Expander,
        lower::generics::ImplTraitLowerFn,
        path::{AssociatedTypeBinding, GenericArg, GenericArgs, GenericArgsParentheses, Path},
    },
    hir::{
        Array, Binding, BindingAnnotation, BindingId, BindingProblems, CaptureBy, ClosureKind,
        Expr, ExprId, Item, Label, LabelId, Literal, MatchArm, Movability, OffsetOf, Pat, PatId,
        RecordFieldPat, RecordLitField, Statement,
        format_args::{
            self, FormatAlignment, FormatArgs, FormatArgsPiece, FormatArgument, FormatArgumentKind,
            FormatArgumentsCollector, FormatCount, FormatDebugHex, FormatOptions,
            FormatPlaceholder, FormatSign, FormatTrait,
        },
        generics::GenericParams,
    },
    item_scope::BuiltinShadowMode,
    item_tree::FieldsShape,
    lang_item::LangItem,
    nameres::{DefMap, LocalDefMap, MacroSubNs, block_def_map},
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
) -> (Body, BodySourceMap) {
    // We cannot leave the root span map empty and let any identifier from it be treated as root,
    // because when inside nested macros `SyntaxContextId`s from the outer macro will be interleaved
    // with the inner macro, and that will cause confusion because they won't be the same as `ROOT`
    // even though they should be the same. Also, when the body comes from multiple expansions, their
    // hygiene is different.

    let mut self_param = None;
    let mut source_map_self_param = None;
    let mut params = vec![];
    let mut collector = ExprCollector::new(db, module, current_file_id);

    let skip_body = match owner {
        DefWithBodyId::FunctionId(it) => db.attrs(it.into()),
        DefWithBodyId::StaticId(it) => db.attrs(it.into()),
        DefWithBodyId::ConstId(it) => db.attrs(it.into()),
        DefWithBodyId::VariantId(it) => db.attrs(it.into()),
    }
    .rust_analyzer_tool()
    .any(|attr| *attr.path() == tool_path![skip]);
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
                self_param = Some(binding_id);
                source_map_self_param =
                    Some(collector.expander.in_file(AstPtr::new(&self_param_syn)));
            }
            let count = param_list.params().filter(|it| collector.check_cfg(it)).count();
            params = (0..count).map(|_| collector.missing_pat()).collect();
        };
        let body_expr = collector.missing_expr();
        return (
            Body {
                store: collector.store.finish(),
                params: params.into_boxed_slice(),
                self_param,
                body_expr,
            },
            BodySourceMap { self_param: source_map_self_param, store: collector.source_map },
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
            self_param = Some(binding_id);
            source_map_self_param = Some(collector.expander.in_file(AstPtr::new(&self_param_syn)));
        }

        for param in param_list.params() {
            if collector.check_cfg(&param) {
                let param_pat = collector.collect_pat_top(param.pat());
                params.push(param_pat);
            }
        }
    };

    let body_expr = collector.collect(
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
    );

    (
        Body {
            store: collector.store.finish(),
            params: params.into_boxed_slice(),
            self_param,
            body_expr,
        },
        BodySourceMap { self_param: source_map_self_param, store: collector.source_map },
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
    (expr_collector.store.finish(), expr_collector.source_map, type_ref)
}

pub(crate) fn lower_generic_params(
    db: &dyn DefDatabase,
    module: ModuleId,
    def: GenericDefId,
    file_id: HirFileId,
    param_list: Option<ast::GenericParamList>,
    where_clause: Option<ast::WhereClause>,
) -> (Arc<ExpressionStore>, Arc<GenericParams>, ExpressionStoreSourceMap) {
    let mut expr_collector = ExprCollector::new(db, module, file_id);
    let mut collector = generics::GenericParamsCollector::new(def);
    collector.lower(&mut expr_collector, param_list, where_clause);
    let params = collector.finish();
    (Arc::new(expr_collector.store.finish()), params, expr_collector.source_map)
}

pub(crate) fn lower_impl(
    db: &dyn DefDatabase,
    module: ModuleId,
    impl_syntax: InFile<ast::Impl>,
    impl_id: ImplId,
) -> (ExpressionStore, ExpressionStoreSourceMap, TypeRefId, Option<TraitRef>, Arc<GenericParams>) {
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
    (expr_collector.store.finish(), expr_collector.source_map, self_ty, trait_, params)
}

pub(crate) fn lower_trait(
    db: &dyn DefDatabase,
    module: ModuleId,
    trait_syntax: InFile<ast::Trait>,
    trait_id: TraitId,
) -> (ExpressionStore, ExpressionStoreSourceMap, Arc<GenericParams>) {
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
    (expr_collector.store.finish(), expr_collector.source_map, params)
}

pub(crate) fn lower_trait_alias(
    db: &dyn DefDatabase,
    module: ModuleId,
    trait_syntax: InFile<ast::TraitAlias>,
    trait_id: TraitAliasId,
) -> (ExpressionStore, ExpressionStoreSourceMap, Arc<GenericParams>) {
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
    (expr_collector.store.finish(), expr_collector.source_map, params)
}

pub(crate) fn lower_type_alias(
    db: &dyn DefDatabase,
    module: ModuleId,
    alias: InFile<ast::TypeAlias>,
    type_alias_id: TypeAliasId,
) -> (
    ExpressionStore,
    ExpressionStoreSourceMap,
    Arc<GenericParams>,
    Box<[TypeBound]>,
    Option<TypeRefId>,
) {
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
    (expr_collector.store.finish(), expr_collector.source_map, params, bounds, type_ref)
}

pub(crate) fn lower_function(
    db: &dyn DefDatabase,
    module: ModuleId,
    fn_: InFile<ast::Fn>,
    function_id: FunctionId,
) -> (
    ExpressionStore,
    ExpressionStoreSourceMap,
    Arc<GenericParams>,
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

    let return_type = if fn_.value.async_token().is_some() {
        let path = hir_expand::mod_path::path![core::future::Future];
        let mut generic_args: Vec<_> =
            std::iter::repeat_n(None, path.segments().len() - 1).collect();
        let binding = AssociatedTypeBinding {
            name: Name::new_symbol_root(sym::Output),
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
    (
        expr_collector.store.finish(),
        expr_collector.source_map,
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
    expander: Expander,
    def_map: &'db DefMap,
    local_def_map: &'db LocalDefMap,
    module: ModuleId,
    pub store: ExpressionStoreBuilder,
    pub(crate) source_map: ExpressionStoreSourceMap,

    // state stuff
    // Prevent nested impl traits like `impl Foo<impl Bar>`.
    outer_impl_trait: bool,

    is_lowering_coroutine: bool,

    /// Legacy (`macro_rules!`) macros can have multiple definitions and shadow each other,
    /// and we need to find the current definition. So we track the number of definitions we saw.
    current_block_legacy_macro_defs_count: FxHashMap<Name, usize>,

    current_try_block_label: Option<LabelId>,

    label_ribs: Vec<LabelRib>,
    current_binding_owner: Option<ExprId>,

    awaitable_context: Option<Awaitable>,
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

impl ExprCollector<'_> {
    pub fn new(
        db: &dyn DefDatabase,
        module: ModuleId,
        current_file_id: HirFileId,
    ) -> ExprCollector<'_> {
        let (def_map, local_def_map) = module.local_def_map(db);
        let expander = Expander::new(db, current_file_id, def_map);
        ExprCollector {
            db,
            cfg_options: module.krate().cfg_options(db),
            module,
            def_map,
            local_def_map,
            source_map: ExpressionStoreSourceMap::default(),
            store: ExpressionStoreBuilder::default(),
            expander,
            current_try_block_label: None,
            is_lowering_coroutine: false,
            label_ribs: Vec::new(),
            current_binding_owner: None,
            awaitable_context: None,
            current_block_legacy_macro_defs_count: FxHashMap::default(),
            outer_impl_trait: false,
        }
    }

    #[inline]
    pub(crate) fn span_map(&self) -> SpanMapRef<'_> {
        self.expander.span_map()
    }

    pub fn lower_lifetime_ref(&mut self, lifetime: ast::Lifetime) -> LifetimeRefId {
        // FIXME: Keyword check?
        let lifetime_ref = match &*lifetime.text() {
            "" | "'" => LifetimeRef::Error,
            "'static" => LifetimeRef::Static,
            "'_" => LifetimeRef::Placeholder,
            text => LifetimeRef::Named(Name::new_lifetime(text)),
        };
        self.alloc_lifetime_ref(lifetime_ref, AstPtr::new(&lifetime))
    }

    pub fn lower_lifetime_ref_opt(&mut self, lifetime: Option<ast::Lifetime>) -> LifetimeRefId {
        match lifetime {
            Some(lifetime) => self.lower_lifetime_ref(lifetime),
            None => self.alloc_lifetime_ref_desugared(LifetimeRef::Placeholder),
        }
    }

    /// Converts an `ast::TypeRef` to a `hir::TypeRef`.
    pub fn lower_type_ref(
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
                fn lower_abi(abi: ast::Abi) -> Symbol {
                    match abi.abi_string() {
                        Some(tok) => Symbol::intern(tok.text_without_quotes()),
                        // `extern` default to be `extern "C"`.
                        _ => sym::C,
                    }
                }

                let abi = inner.abi().map(lower_abi);
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
            ast::Type::MacroType(mt) => match mt.macro_call() {
                Some(mcall) => {
                    let macro_ptr = AstPtr::new(&mcall);
                    let src = self.expander.in_file(AstPtr::new(&node));
                    let id = self.collect_macro_call(mcall, macro_ptr, true, |this, expansion| {
                        this.lower_type_ref_opt(expansion, impl_trait_lower_fn)
                    });
                    self.source_map.types_map.insert(src, id);
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
        self.source_map.types_map_back.insert(id, ptr);
        self.source_map.types_map.insert(ptr, id);
        id
    }

    fn alloc_lifetime_ref(
        &mut self,
        lifetime_ref: LifetimeRef,
        node: LifetimePtr,
    ) -> LifetimeRefId {
        let id = self.store.lifetimes.alloc(lifetime_ref);
        let ptr = self.expander.in_file(node);
        self.source_map.lifetime_map_back.insert(id, ptr);
        self.source_map.lifetime_map.insert(ptr, id);
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
    pub fn lower_generic_args_from_fn_path(
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

    fn collect(&mut self, expr: Option<ast::Expr>, awaitable: Awaitable) -> ExprId {
        self.awaitable_context.replace(awaitable);
        self.with_label_rib(RibKind::Closure, |this| {
            if awaitable == Awaitable::Yes {
                match expr {
                    Some(e) => {
                        let syntax_ptr = AstPtr::new(&e);
                        let expr = this.collect_expr(e);
                        this.alloc_expr_desugared_with_ptr(
                            Expr::Async { id: None, statements: Box::new([]), tail: Some(expr) },
                            syntax_ptr,
                        )
                    }
                    None => this.missing_expr(),
                }
            } else {
                this.collect_expr_opt(expr)
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
        match node.kind() {
            ast::TypeBoundKind::PathType(path_type) => {
                let m = match node.question_mark_token() {
                    Some(_) => TraitBoundModifier::Maybe,
                    None => TraitBoundModifier::None,
                };
                self.lower_path_type(&path_type, impl_trait_lower_fn)
                    .map(|p| {
                        TypeBound::Path(self.alloc_path(p, AstPtr::new(&path_type).upcast()), m)
                    })
                    .unwrap_or(TypeBound::Error)
            }
            ast::TypeBoundKind::ForType(for_type) => {
                let lt_refs = match for_type.generic_param_list() {
                    Some(gpl) => gpl
                        .lifetime_params()
                        .flat_map(|lp| lp.lifetime().map(|lt| Name::new_lifetime(&lt.text())))
                        .collect(),
                    None => ThinVec::default(),
                };
                let path = for_type.ty().and_then(|ty| match &ty {
                    ast::Type::PathType(path_type) => {
                        self.lower_path_type(path_type, impl_trait_lower_fn).map(|p| (p, ty))
                    }
                    _ => None,
                });
                match path {
                    Some((p, ty)) => {
                        TypeBound::ForLifetime(lt_refs, self.alloc_path(p, AstPtr::new(&ty)))
                    }
                    None => TypeBound::Error,
                }
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
        ConstRef { expr: self.collect_expr_opt(arg.and_then(|it| it.expr())) }
    }

    fn lower_const_arg(&mut self, arg: ast::ConstArg) -> ConstRef {
        ConstRef { expr: self.collect_expr_opt(arg.expr()) }
    }

    fn collect_expr(&mut self, expr: ast::Expr) -> ExprId {
        self.maybe_collect_expr(expr).unwrap_or_else(|| self.missing_expr())
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
                Some(ast::BlockModifier::Try(_)) => self.desugar_try_block(e),
                Some(ast::BlockModifier::Unsafe(_)) => {
                    self.collect_block_(e, |id, statements, tail| Expr::Unsafe {
                        id,
                        statements,
                        tail,
                    })
                }
                Some(ast::BlockModifier::Label(label)) => {
                    let label_hygiene = self.hygiene_id_for(label.syntax().text_range());
                    let label_id = self.collect_label(label);
                    self.with_labeled_rib(label_id, label_hygiene, |this| {
                        this.collect_block_(e, |id, statements, tail| Expr::Block {
                            id,
                            statements,
                            tail,
                            label: Some(label_id),
                        })
                    })
                }
                Some(ast::BlockModifier::Async(_)) => {
                    self.with_label_rib(RibKind::Closure, |this| {
                        this.with_awaitable_block(Awaitable::Yes, |this| {
                            this.collect_block_(e, |id, statements, tail| Expr::Async {
                                id,
                                statements,
                                tail,
                            })
                        })
                    })
                }
                Some(ast::BlockModifier::Const(_)) => {
                    self.with_label_rib(RibKind::Constant, |this| {
                        this.with_awaitable_block(Awaitable::No("constant block"), |this| {
                            let (result_expr_id, prev_binding_owner) =
                                this.initialize_binding_owner(syntax_ptr);
                            let inner_expr = this.collect_block(e);
                            this.store.exprs[result_expr_id] = Expr::Const(inner_expr);
                            this.current_binding_owner = prev_binding_owner;
                            result_expr_id
                        })
                    })
                }
                // FIXME
                Some(ast::BlockModifier::AsyncGen(_)) => {
                    self.with_awaitable_block(Awaitable::Yes, |this| this.collect_block(e))
                }
                Some(ast::BlockModifier::Gen(_)) => self
                    .with_awaitable_block(Awaitable::No("non-async gen block"), |this| {
                        this.collect_block(e)
                    }),
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
                // FIXME: Remove this once we drop support for <1.86, https://github.com/rust-lang/rust/commit/ac9cb908ac4301dfc25e7a2edee574320022ae2c
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
                    self.source_map.diagnostics.push(e);
                    None
                });
                self.alloc_expr(Expr::Continue { label }, syntax_ptr)
            }
            ast::Expr::BreakExpr(e) => {
                let label = self.resolve_label(e.lifetime()).unwrap_or_else(|e| {
                    self.source_map.diagnostics.push(e);
                    None
                });
                let expr = e.expr().map(|e| self.collect_expr(e));
                self.alloc_expr(Expr::Break { expr, label }, syntax_ptr)
            }
            ast::Expr::ParenExpr(e) => {
                let inner = self.collect_expr_opt(e.expr());
                // make the paren expr point to the inner expression as well for IDE resolution
                let src = self.expander.in_file(syntax_ptr);
                self.source_map.expr_map.insert(src, inner.into());
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
                    .and_then(|path| self.lower_path(path, &mut Self::impl_trait_error_allocator))
                    .map(Box::new);
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
                            self.source_map.field_map_back.insert(expr, src);
                            Some(RecordLitField { name, expr })
                        })
                        .collect();
                    let spread = nfl.spread().map(|s| self.collect_expr(s));
                    Expr::RecordLit { path, fields, spread }
                } else {
                    Expr::RecordLit { path, fields: Box::default(), spread: None }
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
                    self.source_map.diagnostics.push(
                        ExpressionStoreDiagnostics::AwaitOutsideOfAsync {
                            node: self.expander.in_file(AstPtr::new(&e)),
                            location: location.to_string(),
                        },
                    );
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
                let (result_expr_id, prev_binding_owner) =
                    this.initialize_binding_owner(syntax_ptr);
                let mut args = Vec::new();
                let mut arg_types = Vec::new();
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
                let prev_try_block_label = this.current_try_block_label.take();

                let awaitable = if e.async_token().is_some() {
                    Awaitable::Yes
                } else {
                    Awaitable::No("non-async closure")
                };
                let body =
                    this.with_awaitable_block(awaitable, |this| this.collect_expr_opt(e.body()));

                let closure_kind = if this.is_lowering_coroutine {
                    let movability = if e.static_token().is_some() {
                        Movability::Static
                    } else {
                        Movability::Movable
                    };
                    ClosureKind::Coroutine(movability)
                } else if e.async_token().is_some() {
                    ClosureKind::Async
                } else {
                    ClosureKind::Closure
                };
                let capture_by =
                    if e.move_token().is_some() { CaptureBy::Value } else { CaptureBy::Ref };
                this.is_lowering_coroutine = prev_is_lowering_coroutine;
                this.current_binding_owner = prev_binding_owner;
                this.current_try_block_label = prev_try_block_label;
                this.store.exprs[result_expr_id] = Expr::Closure {
                    args: args.into(),
                    arg_types: arg_types.into(),
                    ret_type,
                    body,
                    closure_kind,
                    capture_by,
                };
                result_expr_id
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
                        let elements = e.map(|expr| self.collect_expr(expr)).collect();
                        self.alloc_expr(Expr::Array(Array::ElementList { elements }), syntax_ptr)
                    }
                    ArrayExprKind::Repeat { initializer, repeat } => {
                        let initializer = self.collect_expr_opt(initializer);
                        let repeat = self.with_label_rib(RibKind::Constant, |this| {
                            if let Some(repeat) = repeat {
                                let syntax_ptr = AstPtr::new(&repeat);
                                this.collect_as_a_binding_owner_bad(
                                    |this| this.collect_expr(repeat),
                                    syntax_ptr,
                                )
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
                        self.source_map.expr_map.insert(src, id.into());
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
            self.source_map.pat_map_back.insert(id, src);
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
                if let Some(ast::Expr::RangeExpr(range)) = e.expr() {
                    if range.is_range_full() {
                        return Some(self.alloc_pat_from_expr(
                            Pat::Tuple { args: Box::default(), ellipsis: Some(0) },
                            syntax_ptr,
                        ));
                    }
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
                    .and_then(|path| self.lower_path(path, &mut Self::impl_trait_error_allocator))
                    .map(Box::new);
                let (ellipsis, args) = collect_tuple(self, e.arg_list()?.args());
                self.alloc_pat_from_expr(Pat::TupleStruct { path, args, ellipsis }, syntax_ptr)
            }
            ast::Expr::PathExpr(e) => {
                let (path, hygiene) = self
                    .collect_expr_path(e.clone())
                    .map(|(path, hygiene)| (Pat::Path(path), hygiene))
                    .unwrap_or((Pat::Missing, HygieneId::ROOT));
                let pat_id = self.alloc_pat_from_expr(path, syntax_ptr);
                if !hygiene.is_root() {
                    self.store.ident_hygiene.insert(pat_id.into(), hygiene);
                }
                pat_id
            }
            ast::Expr::MacroExpr(e) => {
                let e = e.macro_call()?;
                let macro_ptr = AstPtr::new(&e);
                let src = self.expander.in_file(AstPtr::new(expr));
                let id = self.collect_macro_call(e, macro_ptr, true, |this, expansion| {
                    this.collect_expr_as_pat_opt(expansion)
                });
                self.source_map.expr_map.insert(src, id.into());
                id
            }
            ast::Expr::RecordExpr(e) => {
                let path = e
                    .path()
                    .and_then(|path| self.lower_path(path, &mut Self::impl_trait_error_allocator))
                    .map(Box::new);
                let record_field_list = e.record_expr_field_list()?;
                let ellipsis = record_field_list.dotdot_token().is_some();
                // FIXME: Report an error here if `record_field_list.spread().is_some()`.
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
                        self.source_map.pat_field_map_back.insert(pat, src);
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
                            this.source_map.pat_map_back.insert(pat, src);
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

    fn initialize_binding_owner(
        &mut self,
        syntax_ptr: AstPtr<ast::Expr>,
    ) -> (ExprId, Option<ExprId>) {
        let result_expr_id = self.alloc_expr(Expr::Missing, syntax_ptr);
        let prev_binding_owner = self.current_binding_owner.take();
        self.current_binding_owner = Some(result_expr_id);

        (result_expr_id, prev_binding_owner)
    }

    /// FIXME: This function is bad. It will produce a dangling `Missing` expr which wastes memory. Currently
    /// it is used only for const blocks and repeat expressions, which are also hacky and ideally should have
    /// their own body. Don't add more usage for this function so that we can remove this function after
    /// separating those bodies.
    fn collect_as_a_binding_owner_bad(
        &mut self,
        job: impl FnOnce(&mut ExprCollector<'_>) -> ExprId,
        syntax_ptr: AstPtr<ast::Expr>,
    ) -> ExprId {
        let (id, prev_owner) = self.initialize_binding_owner(syntax_ptr);
        let tmp = job(self);
        self.store.exprs[id] = mem::replace(&mut self.store.exprs[tmp], Expr::Missing);
        self.current_binding_owner = prev_owner;
        id
    }

    /// Desugar `try { <stmts>; <expr> }` into `'<new_label>: { <stmts>; ::std::ops::Try::from_output(<expr>) }`,
    /// `try { <stmts>; }` into `'<new_label>: { <stmts>; ::std::ops::Try::from_output(()) }`
    /// and save the `<new_label>` to use it as a break target for desugaring of the `?` operator.
    fn desugar_try_block(&mut self, e: BlockExpr) -> ExprId {
        let try_from_output = self.lang_path(LangItem::TryTraitFromOutput);
        let label = self.alloc_label_desugared(Label {
            name: Name::generate_new_name(self.store.labels.len()),
        });
        let old_label = self.current_try_block_label.replace(label);

        let ptr = AstPtr::new(&e).upcast();
        let (btail, expr_id) = self.with_labeled_rib(label, HygieneId::ROOT, |this| {
            let mut btail = None;
            let block = this.collect_block_(e, |id, statements, tail| {
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
        self.current_try_block_label = old_label;
        expr_id
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
        let into_iter_fn = self.lang_path(LangItem::IntoIterIntoIter);
        let iter_next_fn = self.lang_path(LangItem::IteratorNext);
        let option_some = self.lang_path(LangItem::OptionSome);
        let option_none = self.lang_path(LangItem::OptionNone);
        let head = self.collect_expr_opt(e.iterable());
        let into_iter_fn_expr =
            self.alloc_expr(into_iter_fn.map_or(Expr::Missing, Expr::Path), syntax_ptr);
        let iterator = self.alloc_expr(
            Expr::Call { callee: into_iter_fn_expr, args: Box::new([head]) },
            syntax_ptr,
        );
        let none_arm = MatchArm {
            pat: self.alloc_pat_desugared(option_none.map_or(Pat::Missing, Pat::Path)),
            guard: None,
            expr: self.alloc_expr(Expr::Break { expr: None, label: None }, syntax_ptr),
        };
        let some_pat = Pat::TupleStruct {
            path: option_some.map(Box::new),
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
        let iter_name = Name::generate_new_name(self.store.exprs.len());
        let iter_expr = self.alloc_expr(Expr::Path(Path::from(iter_name.clone())), syntax_ptr);
        let iter_expr_mut = self.alloc_expr(
            Expr::Ref { expr: iter_expr, rawness: Rawness::Ref, mutability: Mutability::Mut },
            syntax_ptr,
        );
        let iter_next_fn_expr =
            self.alloc_expr(iter_next_fn.map_or(Expr::Missing, Expr::Path), syntax_ptr);
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
    ///         break 'catch_target Try::from_residual(residual),
    ///         // Otherwise:
    ///         return Try::from_residual(residual),
    /// }
    /// ```
    fn collect_try_operator(&mut self, syntax_ptr: AstPtr<ast::Expr>, e: ast::TryExpr) -> ExprId {
        let try_branch = self.lang_path(LangItem::TryTraitBranch);
        let cf_continue = self.lang_path(LangItem::ControlFlowContinue);
        let cf_break = self.lang_path(LangItem::ControlFlowBreak);
        let try_from_residual = self.lang_path(LangItem::TryTraitFromResidual);
        let operand = self.collect_expr_opt(e.expr());
        let try_branch = self.alloc_expr(try_branch.map_or(Expr::Missing, Expr::Path), syntax_ptr);
        let expr = self
            .alloc_expr(Expr::Call { callee: try_branch, args: Box::new([operand]) }, syntax_ptr);
        let continue_name = Name::generate_new_name(self.store.bindings.len());
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
                path: cf_continue.map(Box::new),
                args: Box::new([continue_bpat]),
                ellipsis: None,
            }),
            guard: None,
            expr: self.alloc_expr(Expr::Path(Path::from(continue_name)), syntax_ptr),
        };
        let break_name = Name::generate_new_name(self.store.bindings.len());
        let break_binding =
            self.alloc_binding(break_name.clone(), BindingAnnotation::Unannotated, HygieneId::ROOT);
        let break_bpat = self.alloc_pat_desugared(Pat::Bind { id: break_binding, subpat: None });
        self.add_definition_to_binding(break_binding, break_bpat);
        let break_arm = MatchArm {
            pat: self.alloc_pat_desugared(Pat::TupleStruct {
                path: cf_break.map(Box::new),
                args: Box::new([break_bpat]),
                ellipsis: None,
            }),
            guard: None,
            expr: {
                let it = self.alloc_expr(Expr::Path(Path::from(break_name)), syntax_ptr);
                let callee = self
                    .alloc_expr(try_from_residual.map_or(Expr::Missing, Expr::Path), syntax_ptr);
                let result =
                    self.alloc_expr(Expr::Call { callee, args: Box::new([it]) }, syntax_ptr);
                self.alloc_expr(
                    match self.current_try_block_label {
                        Some(label) => Expr::Break { expr: Some(result), label: Some(label) },
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
        let module = self.module.local_id;

        let block_call = self.def_map.modules[self.module.local_id].scope.macro_invoc(
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
                            module,
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
                    self.module.krate(),
                    resolver,
                    &mut |ptr, call| {
                        _ = self.source_map.expansions.insert(ptr.map(|(it, _)| it), call);
                    },
                )
            }
        };

        let res = match res {
            Ok(res) => res,
            Err(UnresolvedMacro { path }) => {
                if record_diagnostics {
                    self.source_map.diagnostics.push(
                        ExpressionStoreDiagnostics::UnresolvedMacroCall {
                            node: self.expander.in_file(syntax_ptr),
                            path,
                        },
                    );
                }
                return collector(self, None);
            }
        };
        if record_diagnostics {
            if let Some(err) = res.err {
                self.source_map
                    .diagnostics
                    .push(ExpressionStoreDiagnostics::MacroError { node: macro_call_ptr, err });
            }
        }

        match res.value {
            Some((mark, expansion)) => {
                // Keep collecting even with expansion errors so we can provide completions and
                // other services in incomplete macro expressions.
                if let Some(macro_file) = self.expander.current_file_id().macro_file() {
                    self.source_map.expansions.insert(macro_call_ptr, macro_file);
                }

                if record_diagnostics {
                    // FIXME: Report parse errors here
                }

                let id = collector(self, expansion.map(|it| it.tree()));
                self.expander.exit(mark);
                id
            }
            None => collector(self, None),
        }
    }

    fn collect_expr_opt(&mut self, expr: Option<ast::Expr>) -> ExprId {
        match expr {
            Some(expr) => self.collect_expr(expr),
            None => self.missing_expr(),
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
            self.source_map.expr_map.insert(src, tail.into());
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
                let macro_id = self.def_map.modules[DefMap::ROOT].scope.get(&name).take_macros();
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
                let macro_id = self.def_map.modules[DefMap::ROOT]
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
        self.collect_block_(block, |id, statements, tail| Expr::Block {
            id,
            statements,
            tail,
            label: None,
        })
    }

    fn collect_block_(
        &mut self,
        block: ast::BlockExpr,
        mk_block: impl FnOnce(Option<BlockId>, Box<[Statement]>, Option<ExprId>) -> Expr,
    ) -> ExprId {
        let block_id = self.expander.ast_id_map().ast_id_for_block(&block).map(|file_local_id| {
            let ast_id = self.expander.in_file(file_local_id);
            self.db.intern_block(BlockLoc { ast_id, module: self.module })
        });

        let (module, def_map) =
            match block_id.map(|block_id| (block_def_map(self.db, block_id), block_id)) {
                Some((def_map, block_id)) => {
                    self.store.block_scopes.push(block_id);
                    (def_map.module_id(DefMap::ROOT), def_map)
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
        let expr_id = self
            .alloc_expr(mk_block(block_id, statements.into_boxed_slice(), tail), syntax_node_ptr);

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
                        self.module.local_id,
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
                            if self.db.variant_fields(variant.into()).shape != FieldsShape::Record =>
                        {
                            (None, Pat::Path(name.into()))
                        }
                        Some(ModuleDefId::AdtId(AdtId::StructId(s)))
                        // FIXME: This can cause a cycle if the user is writing invalid code
                            if self.db.struct_signature(s).shape != FieldsShape::Record =>
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
                    .and_then(|path| self.lower_path(path, &mut Self::impl_trait_error_allocator))
                    .map(Box::new);
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
                    .and_then(|path| self.lower_path(path, &mut Self::impl_trait_error_allocator))
                    .map(Box::new);
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
                        self.source_map.pat_field_map_back.insert(pat, src);
                        Some(RecordFieldPat { name, pat })
                    })
                    .collect();

                let ellipsis = record_pat_field_list.rest_pat().is_some();

                Pat::Record { path, args, ellipsis }
            }
            ast::Pat::SlicePat(p) => {
                let SlicePatComponents { prefix, slice, suffix } = p.components();

                // FIXME properly handle `RestPat`
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
            ast::Pat::RestPat(_) => {
                // `RestPat` requires special handling and should not be mapped
                // to a Pat. Here we are using `Pat::Missing` as a fallback for
                // when `RestPat` is mapped to `Pat`, which can easily happen
                // when the source code being analyzed has a malformed pattern
                // which includes `..` in a place where it isn't valid.

                Pat::Missing
            }
            ast::Pat::BoxPat(boxpat) => {
                let inner = self.collect_pat_opt(boxpat.pat(), binding_list);
                Pat::Box { inner }
            }
            ast::Pat::ConstBlockPat(const_block_pat) => {
                if let Some(block) = const_block_pat.block_expr() {
                    let expr_id = self.with_label_rib(RibKind::Constant, |this| {
                        let syntax_ptr = AstPtr::new(&block.clone().into());
                        this.collect_as_a_binding_owner_bad(
                            |this| this.collect_block(block),
                            syntax_ptr,
                        )
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
                    self.source_map.pat_map.insert(src, pat.into());
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
                Pat::Range { start, end }
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
                        self.source_map.pat_map.insert(src, pat.into());
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

    // endregion: patterns

    /// Returns `None` (and emits diagnostics) when `owner` if `#[cfg]`d out, and `Some(())` when
    /// not.
    fn check_cfg(&mut self, owner: &dyn ast::HasAttrs) -> bool {
        let enabled = self.expander.is_cfg_enabled(self.db, owner, self.cfg_options);
        match enabled {
            Ok(()) => true,
            Err(cfg) => {
                self.source_map.diagnostics.push(ExpressionStoreDiagnostics::InactiveCode {
                    node: self.expander.in_file(SyntaxNodePtr::new(owner.syntax())),
                    cfg,
                    opts: self.cfg_options.clone(),
                });
                false
            }
        }
    }

    fn add_definition_to_binding(&mut self, binding_id: BindingId, pat_id: PatId) {
        self.source_map.binding_definitions.entry(binding_id).or_default().push(pat_id);
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
            hygiene_id.lookup().outer_expn(self.db).map(|expansion| {
                let expansion = self.db.lookup_intern_macro_call(expansion.into());
                (hygiene_id.lookup().parent(self.db), expansion.def)
            })
        };
        let name = Name::new_lifetime(&lifetime.text());

        for (rib_idx, rib) in self.label_ribs.iter().enumerate().rev() {
            match &rib.kind {
                RibKind::Normal(label_name, id, label_hygiene) => {
                    if *label_name == name && *label_hygiene == hygiene_id {
                        return if self.is_label_valid_from_rib(rib_idx) {
                            Ok(Some(*id))
                        } else {
                            Err(ExpressionStoreDiagnostics::UnreachableLabel {
                                name,
                                node: self.expander.in_file(AstPtr::new(&lifetime)),
                            })
                        };
                    }
                }
                RibKind::MacroDef(macro_id) => {
                    if let Some((parent_ctx, label_macro_id)) = hygiene_info {
                        if label_macro_id == **macro_id {
                            // A macro is allowed to refer to labels from before its declaration.
                            // Therefore, if we got to the rib of its declaration, give up its hygiene
                            // and use its parent expansion.

                            hygiene_id =
                                HygieneId::new(parent_ctx.opaque_and_semitransparent(self.db));
                            hygiene_info = parent_ctx.outer_expn(self.db).map(|expansion| {
                                let expansion = self.db.lookup_intern_macro_call(expansion.into());
                                (parent_ctx.parent(self.db), expansion.def)
                            });
                        }
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

    // region: format
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

    fn collect_format_args(
        &mut self,
        f: ast::FormatArgsExpr,
        syntax_ptr: AstPtr<ast::Expr>,
    ) -> ExprId {
        let mut args = FormatArgumentsCollector::default();
        f.args().for_each(|arg| {
            args.add(FormatArgument {
                kind: match arg.name() {
                    Some(name) => FormatArgumentKind::Named(name.as_name()),
                    None => FormatArgumentKind::Normal,
                },
                expr: self.collect_expr_opt(arg.expr()),
            });
        });
        let template = f.template();
        let fmt_snippet = template.as_ref().and_then(|it| match it {
            ast::Expr::Literal(literal) => match literal.kind() {
                ast::LiteralKind::String(s) => Some(s.text().to_owned()),
                _ => None,
            },
            _ => None,
        });
        let mut mappings = vec![];
        let (fmt, hygiene) = match template.and_then(|template| {
            self.expand_macros_to_string(template.clone()).map(|it| (it, template))
        }) {
            Some(((s, is_direct_literal), template)) => {
                let call_ctx = self.expander.call_syntax_ctx();
                let hygiene = self.hygiene_id_for(s.syntax().text_range());
                let fmt = format_args::parse(
                    &s,
                    fmt_snippet,
                    args,
                    is_direct_literal,
                    |name, range| {
                        let expr_id = self.alloc_expr_desugared(Expr::Path(Path::from(name)));
                        if let Some(range) = range {
                            self.source_map
                                .template_map
                                .get_or_insert_with(Default::default)
                                .implicit_capture_to_source
                                .insert(
                                    expr_id,
                                    self.expander.in_file((AstPtr::new(&template), range)),
                                );
                        }
                        if !hygiene.is_root() {
                            self.store.ident_hygiene.insert(expr_id.into(), hygiene);
                        }
                        expr_id
                    },
                    |name, span| {
                        if let Some(span) = span {
                            mappings.push((span, name))
                        }
                    },
                    call_ctx,
                );
                (fmt, hygiene)
            }
            None => (
                FormatArgs {
                    template: Default::default(),
                    arguments: args.finish(),
                    orphans: Default::default(),
                },
                HygieneId::ROOT,
            ),
        };

        // Create a list of all _unique_ (argument, format trait) combinations.
        // E.g. "{0} {0:x} {0} {1}" -> [(0, Display), (0, LowerHex), (1, Display)]
        let mut argmap = FxIndexSet::default();
        for piece in fmt.template.iter() {
            let FormatArgsPiece::Placeholder(placeholder) = piece else { continue };
            if let Ok(index) = placeholder.argument.index {
                argmap.insert((index, ArgumentType::Format(placeholder.format_trait)));
            }
        }

        let lit_pieces = fmt
            .template
            .iter()
            .enumerate()
            .filter_map(|(i, piece)| {
                match piece {
                    FormatArgsPiece::Literal(s) => {
                        Some(self.alloc_expr_desugared(Expr::Literal(Literal::String(s.clone()))))
                    }
                    &FormatArgsPiece::Placeholder(_) => {
                        // Inject empty string before placeholders when not already preceded by a literal piece.
                        if i == 0 || matches!(fmt.template[i - 1], FormatArgsPiece::Placeholder(_))
                        {
                            Some(self.alloc_expr_desugared(Expr::Literal(Literal::String(
                                Symbol::empty(),
                            ))))
                        } else {
                            None
                        }
                    }
                }
            })
            .collect();
        let lit_pieces =
            self.alloc_expr_desugared(Expr::Array(Array::ElementList { elements: lit_pieces }));
        let lit_pieces = self.alloc_expr_desugared(Expr::Ref {
            expr: lit_pieces,
            rawness: Rawness::Ref,
            mutability: Mutability::Shared,
        });
        let format_options = {
            // Generate:
            //     &[format_spec_0, format_spec_1, format_spec_2]
            let elements = fmt
                .template
                .iter()
                .filter_map(|piece| {
                    let FormatArgsPiece::Placeholder(placeholder) = piece else { return None };
                    Some(self.make_format_spec(placeholder, &mut argmap))
                })
                .collect();
            let array = self.alloc_expr_desugared(Expr::Array(Array::ElementList { elements }));
            self.alloc_expr_desugared(Expr::Ref {
                expr: array,
                rawness: Rawness::Ref,
                mutability: Mutability::Shared,
            })
        };

        // Assume that rustc version >= 1.89.0 iff lang item `format_arguments` exists
        // but `format_unsafe_arg` does not
        let fmt_args =
            || crate::lang_item::lang_item(self.db, self.module.krate(), LangItem::FormatArguments);
        let fmt_unsafe_arg =
            || crate::lang_item::lang_item(self.db, self.module.krate(), LangItem::FormatUnsafeArg);
        let use_format_args_since_1_89_0 = fmt_args().is_some() && fmt_unsafe_arg().is_none();

        let idx = if use_format_args_since_1_89_0 {
            self.collect_format_args_impl(
                syntax_ptr,
                fmt,
                hygiene,
                argmap,
                lit_pieces,
                format_options,
            )
        } else {
            self.collect_format_args_before_1_89_0_impl(
                syntax_ptr,
                fmt,
                argmap,
                lit_pieces,
                format_options,
            )
        };

        self.source_map
            .template_map
            .get_or_insert_with(Default::default)
            .format_args_to_captures
            .insert(idx, (hygiene, mappings));
        idx
    }

    /// `format_args!` expansion implementation for rustc versions < `1.89.0`
    fn collect_format_args_before_1_89_0_impl(
        &mut self,
        syntax_ptr: AstPtr<ast::Expr>,
        fmt: FormatArgs,
        argmap: FxIndexSet<(usize, ArgumentType)>,
        lit_pieces: ExprId,
        format_options: ExprId,
    ) -> ExprId {
        let arguments = &*fmt.arguments.arguments;

        let args = if arguments.is_empty() {
            let expr = self
                .alloc_expr_desugared(Expr::Array(Array::ElementList { elements: Box::default() }));
            self.alloc_expr_desugared(Expr::Ref {
                expr,
                rawness: Rawness::Ref,
                mutability: Mutability::Shared,
            })
        } else {
            // Generate:
            //     &match (&arg0, &arg1, &…) {
            //         args => [
            //             <core::fmt::Argument>::new_display(args.0),
            //             <core::fmt::Argument>::new_lower_hex(args.1),
            //             <core::fmt::Argument>::new_debug(args.0),
            //             …
            //         ]
            //     }
            let args = argmap
                .iter()
                .map(|&(arg_index, ty)| {
                    let arg = self.alloc_expr_desugared(Expr::Ref {
                        expr: arguments[arg_index].expr,
                        rawness: Rawness::Ref,
                        mutability: Mutability::Shared,
                    });
                    self.make_argument(arg, ty)
                })
                .collect();
            let array =
                self.alloc_expr_desugared(Expr::Array(Array::ElementList { elements: args }));
            self.alloc_expr_desugared(Expr::Ref {
                expr: array,
                rawness: Rawness::Ref,
                mutability: Mutability::Shared,
            })
        };

        // Generate:
        //     <core::fmt::Arguments>::new_v1_formatted(
        //         lit_pieces,
        //         args,
        //         format_options,
        //         unsafe { ::core::fmt::UnsafeArg::new() }
        //     )

        let new_v1_formatted = LangItem::FormatArguments.ty_rel_path(
            self.db,
            self.module.krate(),
            Name::new_symbol_root(sym::new_v1_formatted),
        );
        let unsafe_arg_new = LangItem::FormatUnsafeArg.ty_rel_path(
            self.db,
            self.module.krate(),
            Name::new_symbol_root(sym::new),
        );
        let new_v1_formatted =
            self.alloc_expr_desugared(new_v1_formatted.map_or(Expr::Missing, Expr::Path));

        let unsafe_arg_new =
            self.alloc_expr_desugared(unsafe_arg_new.map_or(Expr::Missing, Expr::Path));
        let unsafe_arg_new =
            self.alloc_expr_desugared(Expr::Call { callee: unsafe_arg_new, args: Box::default() });
        let mut unsafe_arg_new = self.alloc_expr_desugared(Expr::Unsafe {
            id: None,
            statements: Box::new([]),
            tail: Some(unsafe_arg_new),
        });
        if !fmt.orphans.is_empty() {
            unsafe_arg_new = self.alloc_expr_desugared(Expr::Block {
                id: None,
                // We collect the unused expressions here so that we still infer them instead of
                // dropping them out of the expression tree. We cannot store them in the `Unsafe`
                // block because then unsafe blocks within them will get a false "unused unsafe"
                // diagnostic (rustc has a notion of builtin unsafe blocks, but we don't).
                statements: fmt
                    .orphans
                    .into_iter()
                    .map(|expr| Statement::Expr { expr, has_semi: true })
                    .collect(),
                tail: Some(unsafe_arg_new),
                label: None,
            });
        }

        self.alloc_expr(
            Expr::Call {
                callee: new_v1_formatted,
                args: Box::new([lit_pieces, args, format_options, unsafe_arg_new]),
            },
            syntax_ptr,
        )
    }

    /// `format_args!` expansion implementation for rustc versions >= `1.89.0`,
    /// especially since [this PR](https://github.com/rust-lang/rust/pull/140748)
    fn collect_format_args_impl(
        &mut self,
        syntax_ptr: AstPtr<ast::Expr>,
        fmt: FormatArgs,
        hygiene: HygieneId,
        argmap: FxIndexSet<(usize, ArgumentType)>,
        lit_pieces: ExprId,
        format_options: ExprId,
    ) -> ExprId {
        let arguments = &*fmt.arguments.arguments;

        let (let_stmts, args) = if arguments.is_empty() {
            (
                // Generate:
                //     []
                vec![],
                self.alloc_expr_desugared(Expr::Array(Array::ElementList {
                    elements: Box::default(),
                })),
            )
        } else if argmap.len() == 1 && arguments.len() == 1 {
            // Only one argument, so we don't need to make the `args` tuple.
            //
            // Generate:
            //     super let args = [<core::fmt::Arguments>::new_display(&arg)];
            let args = argmap
                .iter()
                .map(|&(arg_index, ty)| {
                    let ref_arg = self.alloc_expr_desugared(Expr::Ref {
                        expr: arguments[arg_index].expr,
                        rawness: Rawness::Ref,
                        mutability: Mutability::Shared,
                    });
                    self.make_argument(ref_arg, ty)
                })
                .collect();
            let args =
                self.alloc_expr_desugared(Expr::Array(Array::ElementList { elements: args }));
            let args_name = Name::new_symbol_root(sym::args);
            let args_binding =
                self.alloc_binding(args_name.clone(), BindingAnnotation::Unannotated, hygiene);
            let args_pat = self.alloc_pat_desugared(Pat::Bind { id: args_binding, subpat: None });
            self.add_definition_to_binding(args_binding, args_pat);
            // TODO: We don't have `super let` yet.
            let let_stmt = Statement::Let {
                pat: args_pat,
                type_ref: None,
                initializer: Some(args),
                else_branch: None,
            };
            (vec![let_stmt], self.alloc_expr_desugared(Expr::Path(Path::from(args_name))))
        } else {
            // Generate:
            //     super let args = (&arg0, &arg1, &...);
            let args_name = Name::new_symbol_root(sym::args);
            let args_binding =
                self.alloc_binding(args_name.clone(), BindingAnnotation::Unannotated, hygiene);
            let args_pat = self.alloc_pat_desugared(Pat::Bind { id: args_binding, subpat: None });
            self.add_definition_to_binding(args_binding, args_pat);
            let elements = arguments
                .iter()
                .map(|arg| {
                    self.alloc_expr_desugared(Expr::Ref {
                        expr: arg.expr,
                        rawness: Rawness::Ref,
                        mutability: Mutability::Shared,
                    })
                })
                .collect();
            let args_tuple = self.alloc_expr_desugared(Expr::Tuple { exprs: elements });
            // TODO: We don't have `super let` yet
            let let_stmt1 = Statement::Let {
                pat: args_pat,
                type_ref: None,
                initializer: Some(args_tuple),
                else_branch: None,
            };

            // Generate:
            //     super let args = [
            //         <core::fmt::Argument>::new_display(args.0),
            //         <core::fmt::Argument>::new_lower_hex(args.1),
            //         <core::fmt::Argument>::new_debug(args.0),
            //         …
            //     ];
            let args = argmap
                .iter()
                .map(|&(arg_index, ty)| {
                    let args_ident_expr =
                        self.alloc_expr_desugared(Expr::Path(args_name.clone().into()));
                    let arg = self.alloc_expr_desugared(Expr::Field {
                        expr: args_ident_expr,
                        name: Name::new_tuple_field(arg_index),
                    });
                    self.make_argument(arg, ty)
                })
                .collect();
            let array =
                self.alloc_expr_desugared(Expr::Array(Array::ElementList { elements: args }));
            let args_binding =
                self.alloc_binding(args_name.clone(), BindingAnnotation::Unannotated, hygiene);
            let args_pat = self.alloc_pat_desugared(Pat::Bind { id: args_binding, subpat: None });
            self.add_definition_to_binding(args_binding, args_pat);
            let let_stmt2 = Statement::Let {
                pat: args_pat,
                type_ref: None,
                initializer: Some(array),
                else_branch: None,
            };
            (vec![let_stmt1, let_stmt2], self.alloc_expr_desugared(Expr::Path(args_name.into())))
        };

        // Generate:
        //     &args
        let args = self.alloc_expr_desugared(Expr::Ref {
            expr: args,
            rawness: Rawness::Ref,
            mutability: Mutability::Shared,
        });

        let call_block = {
            // Generate:
            //     unsafe {
            //         <core::fmt::Arguments>::new_v1_formatted(
            //             lit_pieces,
            //             args,
            //             format_options,
            //         )
            //     }

            let new_v1_formatted = LangItem::FormatArguments.ty_rel_path(
                self.db,
                self.module.krate(),
                Name::new_symbol_root(sym::new_v1_formatted),
            );
            let new_v1_formatted =
                self.alloc_expr_desugared(new_v1_formatted.map_or(Expr::Missing, Expr::Path));
            let args = [lit_pieces, args, format_options];
            let call = self
                .alloc_expr_desugared(Expr::Call { callee: new_v1_formatted, args: args.into() });

            Expr::Unsafe { id: None, statements: Box::default(), tail: Some(call) }
        };

        if !let_stmts.is_empty() {
            // Generate:
            //     {
            //         super let …
            //         super let …
            //         <core::fmt::Arguments>::new_…(…)
            //     }
            let call = self.alloc_expr_desugared(call_block);
            self.alloc_expr(
                Expr::Block {
                    id: None,
                    statements: let_stmts.into(),
                    tail: Some(call),
                    label: None,
                },
                syntax_ptr,
            )
        } else {
            self.alloc_expr(call_block, syntax_ptr)
        }
    }

    /// Generate a hir expression for a format_args placeholder specification.
    ///
    /// Generates
    ///
    /// ```text
    ///     <core::fmt::rt::Placeholder::new(
    ///         …usize, // position
    ///         '…', // fill
    ///         <core::fmt::rt::Alignment>::…, // alignment
    ///         …u32, // flags
    ///         <core::fmt::rt::Count::…>, // width
    ///         <core::fmt::rt::Count::…>, // precision
    ///     )
    /// ```
    fn make_format_spec(
        &mut self,
        placeholder: &FormatPlaceholder,
        argmap: &mut FxIndexSet<(usize, ArgumentType)>,
    ) -> ExprId {
        let position = match placeholder.argument.index {
            Ok(arg_index) => {
                let (i, _) =
                    argmap.insert_full((arg_index, ArgumentType::Format(placeholder.format_trait)));
                self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                    i as u128,
                    Some(BuiltinUint::Usize),
                )))
            }
            Err(_) => self.missing_expr(),
        };
        let &FormatOptions {
            ref width,
            ref precision,
            alignment,
            fill,
            sign,
            alternate,
            zero_pad,
            debug_hex,
        } = &placeholder.format_options;

        let precision_expr = self.make_count(precision, argmap);
        let width_expr = self.make_count(width, argmap);

        if self.module.krate().workspace_data(self.db).is_atleast_187() {
            // These need to match the constants in library/core/src/fmt/rt.rs.
            let align = match alignment {
                Some(FormatAlignment::Left) => 0,
                Some(FormatAlignment::Right) => 1,
                Some(FormatAlignment::Center) => 2,
                None => 3,
            };
            // This needs to match `Flag` in library/core/src/fmt/rt.rs.
            let flags = fill.unwrap_or(' ') as u32
                | ((sign == Some(FormatSign::Plus)) as u32) << 21
                | ((sign == Some(FormatSign::Minus)) as u32) << 22
                | (alternate as u32) << 23
                | (zero_pad as u32) << 24
                | ((debug_hex == Some(FormatDebugHex::Lower)) as u32) << 25
                | ((debug_hex == Some(FormatDebugHex::Upper)) as u32) << 26
                | (width.is_some() as u32) << 27
                | (precision.is_some() as u32) << 28
                | align << 29
                | 1 << 31; // Highest bit always set.
            let flags = self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                flags as u128,
                Some(BuiltinUint::U32),
            )));

            let position =
                RecordLitField { name: Name::new_symbol_root(sym::position), expr: position };
            let flags = RecordLitField { name: Name::new_symbol_root(sym::flags), expr: flags };
            let precision = RecordLitField {
                name: Name::new_symbol_root(sym::precision),
                expr: precision_expr,
            };
            let width =
                RecordLitField { name: Name::new_symbol_root(sym::width), expr: width_expr };
            self.alloc_expr_desugared(Expr::RecordLit {
                path: LangItem::FormatPlaceholder.path(self.db, self.module.krate()).map(Box::new),
                fields: Box::new([position, flags, precision, width]),
                spread: None,
            })
        } else {
            let format_placeholder_new = {
                let format_placeholder_new = LangItem::FormatPlaceholder.ty_rel_path(
                    self.db,
                    self.module.krate(),
                    Name::new_symbol_root(sym::new),
                );
                match format_placeholder_new {
                    Some(path) => self.alloc_expr_desugared(Expr::Path(path)),
                    None => self.missing_expr(),
                }
            };
            // This needs to match `Flag` in library/core/src/fmt/rt.rs.
            let flags: u32 = ((sign == Some(FormatSign::Plus)) as u32)
                | (((sign == Some(FormatSign::Minus)) as u32) << 1)
                | ((alternate as u32) << 2)
                | ((zero_pad as u32) << 3)
                | (((debug_hex == Some(FormatDebugHex::Lower)) as u32) << 4)
                | (((debug_hex == Some(FormatDebugHex::Upper)) as u32) << 5);
            let flags = self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                flags as u128,
                Some(BuiltinUint::U32),
            )));
            let fill = self.alloc_expr_desugared(Expr::Literal(Literal::Char(fill.unwrap_or(' '))));
            let align = {
                let align = LangItem::FormatAlignment.ty_rel_path(
                    self.db,
                    self.module.krate(),
                    match alignment {
                        Some(FormatAlignment::Left) => Name::new_symbol_root(sym::Left),
                        Some(FormatAlignment::Right) => Name::new_symbol_root(sym::Right),
                        Some(FormatAlignment::Center) => Name::new_symbol_root(sym::Center),
                        None => Name::new_symbol_root(sym::Unknown),
                    },
                );
                match align {
                    Some(path) => self.alloc_expr_desugared(Expr::Path(path)),
                    None => self.missing_expr(),
                }
            };
            self.alloc_expr_desugared(Expr::Call {
                callee: format_placeholder_new,
                args: Box::new([position, fill, align, flags, precision_expr, width_expr]),
            })
        }
    }

    /// Generate a hir expression for a format_args Count.
    ///
    /// Generates:
    ///
    /// ```text
    ///     <core::fmt::rt::Count>::Is(…)
    /// ```
    ///
    /// or
    ///
    /// ```text
    ///     <core::fmt::rt::Count>::Param(…)
    /// ```
    ///
    /// or
    ///
    /// ```text
    ///     <core::fmt::rt::Count>::Implied
    /// ```
    fn make_count(
        &mut self,
        count: &Option<FormatCount>,
        argmap: &mut FxIndexSet<(usize, ArgumentType)>,
    ) -> ExprId {
        match count {
            Some(FormatCount::Literal(n)) => {
                let args = self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                    *n as u128,
                    // FIXME: Change this to Some(BuiltinUint::U16) once we drop support for toolchains < 1.88
                    None,
                )));
                let count_is = match LangItem::FormatCount.ty_rel_path(
                    self.db,
                    self.module.krate(),
                    Name::new_symbol_root(sym::Is),
                ) {
                    Some(count_is) => self.alloc_expr_desugared(Expr::Path(count_is)),
                    None => self.missing_expr(),
                };
                self.alloc_expr_desugared(Expr::Call { callee: count_is, args: Box::new([args]) })
            }
            Some(FormatCount::Argument(arg)) => {
                if let Ok(arg_index) = arg.index {
                    let (i, _) = argmap.insert_full((arg_index, ArgumentType::Usize));

                    let args = self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                        i as u128,
                        Some(BuiltinUint::Usize),
                    )));
                    let count_param = match LangItem::FormatCount.ty_rel_path(
                        self.db,
                        self.module.krate(),
                        Name::new_symbol_root(sym::Param),
                    ) {
                        Some(count_param) => self.alloc_expr_desugared(Expr::Path(count_param)),
                        None => self.missing_expr(),
                    };
                    self.alloc_expr_desugared(Expr::Call {
                        callee: count_param,
                        args: Box::new([args]),
                    })
                } else {
                    // FIXME: This drops arg causing it to potentially not be resolved/type checked
                    // when typing?
                    self.missing_expr()
                }
            }
            None => match LangItem::FormatCount.ty_rel_path(
                self.db,
                self.module.krate(),
                Name::new_symbol_root(sym::Implied),
            ) {
                Some(count_param) => self.alloc_expr_desugared(Expr::Path(count_param)),
                None => self.missing_expr(),
            },
        }
    }

    /// Generate a hir expression representing an argument to a format_args invocation.
    ///
    /// Generates:
    ///
    /// ```text
    ///     <core::fmt::Argument>::new_…(arg)
    /// ```
    fn make_argument(&mut self, arg: ExprId, ty: ArgumentType) -> ExprId {
        use ArgumentType::*;
        use FormatTrait::*;

        let new_fn = match LangItem::FormatArgument.ty_rel_path(
            self.db,
            self.module.krate(),
            Name::new_symbol_root(match ty {
                Format(Display) => sym::new_display,
                Format(Debug) => sym::new_debug,
                Format(LowerExp) => sym::new_lower_exp,
                Format(UpperExp) => sym::new_upper_exp,
                Format(Octal) => sym::new_octal,
                Format(Pointer) => sym::new_pointer,
                Format(Binary) => sym::new_binary,
                Format(LowerHex) => sym::new_lower_hex,
                Format(UpperHex) => sym::new_upper_hex,
                Usize => sym::from_usize,
            }),
        ) {
            Some(new_fn) => self.alloc_expr_desugared(Expr::Path(new_fn)),
            None => self.missing_expr(),
        };
        self.alloc_expr_desugared(Expr::Call { callee: new_fn, args: Box::new([arg]) })
    }

    // endregion: format

    fn lang_path(&self, lang: LangItem) -> Option<Path> {
        lang.path(self.db, self.module.krate())
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
    fn alloc_expr(&mut self, expr: Expr, ptr: ExprPtr) -> ExprId {
        let src = self.expander.in_file(ptr);
        let id = self.store.exprs.alloc(expr);
        self.source_map.expr_map_back.insert(id, src.map(AstPtr::wrap_left));
        self.source_map.expr_map.insert(src, id.into());
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
        self.source_map.expr_map_back.insert(id, src.map(AstPtr::wrap_left));
        // We intentionally don't fill this as it could overwrite a non-desugared entry
        // self.source_map.expr_map.insert(src, id);
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
        if let Some(owner) = self.current_binding_owner {
            self.store.binding_owners.insert(binding, owner);
        }
        binding
    }

    fn alloc_pat_from_expr(&mut self, pat: Pat, ptr: ExprPtr) -> PatId {
        let src = self.expander.in_file(ptr);
        let id = self.store.pats.alloc(pat);
        self.source_map.expr_map.insert(src, id.into());
        self.source_map.pat_map_back.insert(id, src.map(AstPtr::wrap_left));
        id
    }

    fn alloc_expr_from_pat(&mut self, expr: Expr, ptr: PatPtr) -> ExprId {
        let src = self.expander.in_file(ptr);
        let id = self.store.exprs.alloc(expr);
        self.source_map.pat_map.insert(src, id.into());
        self.source_map.expr_map_back.insert(id, src.map(AstPtr::wrap_right));
        id
    }

    fn alloc_pat(&mut self, pat: Pat, ptr: PatPtr) -> PatId {
        let src = self.expander.in_file(ptr);
        let id = self.store.pats.alloc(pat);
        self.source_map.pat_map_back.insert(id, src.map(AstPtr::wrap_right));
        self.source_map.pat_map.insert(src, id.into());
        id
    }
    // FIXME: desugared pats don't have ptr, that's wrong and should be fixed somehow.
    fn alloc_pat_desugared(&mut self, pat: Pat) -> PatId {
        self.store.pats.alloc(pat)
    }
    fn missing_pat(&mut self) -> PatId {
        self.store.pats.alloc(Pat::Missing)
    }

    fn alloc_label(&mut self, label: Label, ptr: LabelPtr) -> LabelId {
        let src = self.expander.in_file(ptr);
        let id = self.store.labels.alloc(label);
        self.source_map.label_map_back.insert(id, src);
        self.source_map.label_map.insert(src, id);
        id
    }
    // FIXME: desugared labels don't have ptr, that's wrong and should be fixed somehow.
    fn alloc_label_desugared(&mut self, label: Label) -> LabelId {
        self.store.labels.alloc(label)
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

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum ArgumentType {
    Format(FormatTrait),
    Usize,
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
