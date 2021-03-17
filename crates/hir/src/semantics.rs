//! See `Semantics`.

mod source_to_def;

use std::{cell::RefCell, fmt, iter::successors};

use base_db::{FileId, FileRange};
use hir_def::{
    resolver::{self, HasResolver, Resolver, TypeNs},
    AsMacroCall, FunctionId, TraitId, VariantId,
};
use hir_expand::{hygiene::Hygiene, name::AsName, ExpansionInfo};
use hir_ty::associated_type_shorthand_candidates;
use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};
use syntax::{
    algo::find_node_at_offset,
    ast::{self, GenericParamsOwner, LoopBodyOwner},
    match_ast, AstNode, SyntaxNode, SyntaxNodePtr, SyntaxToken, TextSize,
};

use crate::{
    db::HirDatabase,
    semantics::source_to_def::{ChildContainer, SourceToDefCache, SourceToDefCtx},
    source_analyzer::{resolve_hir_path, SourceAnalyzer},
    Access, AssocItem, Callable, ConstParam, Crate, Field, Function, HirFileId, Impl, InFile,
    Label, LifetimeParam, Local, MacroDef, Module, ModuleDef, Name, Path, ScopeDef, Trait, Type,
    TypeAlias, TypeParam, VariantDef,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathResolution {
    /// An item
    Def(ModuleDef),
    /// A local binding (only value namespace)
    Local(Local),
    /// A generic parameter
    TypeParam(TypeParam),
    ConstParam(ConstParam),
    SelfType(Impl),
    Macro(MacroDef),
    AssocItem(AssocItem),
}

impl PathResolution {
    fn in_type_ns(&self) -> Option<TypeNs> {
        match self {
            PathResolution::Def(ModuleDef::Adt(adt)) => Some(TypeNs::AdtId((*adt).into())),
            PathResolution::Def(ModuleDef::BuiltinType(builtin)) => {
                Some(TypeNs::BuiltinType((*builtin).into()))
            }
            PathResolution::Def(ModuleDef::Const(_))
            | PathResolution::Def(ModuleDef::Variant(_))
            | PathResolution::Def(ModuleDef::Function(_))
            | PathResolution::Def(ModuleDef::Module(_))
            | PathResolution::Def(ModuleDef::Static(_))
            | PathResolution::Def(ModuleDef::Trait(_)) => None,
            PathResolution::Def(ModuleDef::TypeAlias(alias)) => {
                Some(TypeNs::TypeAliasId((*alias).into()))
            }
            PathResolution::Local(_) | PathResolution::Macro(_) | PathResolution::ConstParam(_) => {
                None
            }
            PathResolution::TypeParam(param) => Some(TypeNs::GenericParam((*param).into())),
            PathResolution::SelfType(impl_def) => Some(TypeNs::SelfType((*impl_def).into())),
            PathResolution::AssocItem(AssocItem::Const(_))
            | PathResolution::AssocItem(AssocItem::Function(_)) => None,
            PathResolution::AssocItem(AssocItem::TypeAlias(alias)) => {
                Some(TypeNs::TypeAliasId((*alias).into()))
            }
        }
    }

    /// Returns an iterator over associated types that may be specified after this path (using
    /// `Ty::Assoc` syntax).
    pub fn assoc_type_shorthand_candidates<R>(
        &self,
        db: &dyn HirDatabase,
        mut cb: impl FnMut(TypeAlias) -> Option<R>,
    ) -> Option<R> {
        associated_type_shorthand_candidates(db, self.in_type_ns()?, |_, _, id| cb(id.into()))
    }
}

/// Primary API to get semantic information, like types, from syntax trees.
pub struct Semantics<'db, DB> {
    pub db: &'db DB,
    imp: SemanticsImpl<'db>,
}

pub struct SemanticsImpl<'db> {
    pub db: &'db dyn HirDatabase,
    s2d_cache: RefCell<SourceToDefCache>,
    expansion_info_cache: RefCell<FxHashMap<HirFileId, Option<ExpansionInfo>>>,
    cache: RefCell<FxHashMap<SyntaxNode, HirFileId>>,
}

impl<DB> fmt::Debug for Semantics<'_, DB> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Semantics {{ ... }}")
    }
}

impl<'db, DB: HirDatabase> Semantics<'db, DB> {
    pub fn new(db: &DB) -> Semantics<DB> {
        let impl_ = SemanticsImpl::new(db);
        Semantics { db, imp: impl_ }
    }

    pub fn parse(&self, file_id: FileId) -> ast::SourceFile {
        self.imp.parse(file_id)
    }

    pub fn expand(&self, macro_call: &ast::MacroCall) -> Option<SyntaxNode> {
        self.imp.expand(macro_call)
    }
    pub fn speculative_expand(
        &self,
        actual_macro_call: &ast::MacroCall,
        hypothetical_args: &ast::TokenTree,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, SyntaxToken)> {
        self.imp.speculative_expand(actual_macro_call, hypothetical_args, token_to_map)
    }

    pub fn descend_into_macros(&self, token: SyntaxToken) -> SyntaxToken {
        self.imp.descend_into_macros(token)
    }

    pub fn descend_node_at_offset<N: ast::AstNode>(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> Option<N> {
        self.imp.descend_node_at_offset(node, offset).find_map(N::cast)
    }

    pub fn original_range(&self, node: &SyntaxNode) -> FileRange {
        self.imp.original_range(node)
    }

    pub fn diagnostics_display_range(&self, diagnostics: InFile<SyntaxNodePtr>) -> FileRange {
        self.imp.diagnostics_display_range(diagnostics)
    }

    pub fn token_ancestors_with_macros(
        &self,
        token: SyntaxToken,
    ) -> impl Iterator<Item = SyntaxNode> + '_ {
        token.parent().into_iter().flat_map(move |it| self.ancestors_with_macros(it))
    }
    pub fn ancestors_with_macros(&self, node: SyntaxNode) -> impl Iterator<Item = SyntaxNode> + '_ {
        self.imp.ancestors_with_macros(node)
    }

    pub fn ancestors_at_offset_with_macros(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> impl Iterator<Item = SyntaxNode> + '_ {
        self.imp.ancestors_at_offset_with_macros(node, offset)
    }

    /// Find a AstNode by offset inside SyntaxNode, if it is inside *Macrofile*,
    /// search up until it is of the target AstNode type
    pub fn find_node_at_offset_with_macros<N: AstNode>(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> Option<N> {
        self.imp.ancestors_at_offset_with_macros(node, offset).find_map(N::cast)
    }

    /// Find a AstNode by offset inside SyntaxNode, if it is inside *MacroCall*,
    /// descend it and find again
    pub fn find_node_at_offset_with_descend<N: AstNode>(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> Option<N> {
        if let Some(it) = find_node_at_offset(&node, offset) {
            return Some(it);
        }

        self.imp.descend_node_at_offset(node, offset).find_map(N::cast)
    }

    pub fn resolve_lifetime_param(&self, lifetime: &ast::Lifetime) -> Option<LifetimeParam> {
        self.imp.resolve_lifetime_param(lifetime)
    }

    pub fn resolve_label(&self, lifetime: &ast::Lifetime) -> Option<Label> {
        self.imp.resolve_label(lifetime)
    }

    pub fn type_of_expr(&self, expr: &ast::Expr) -> Option<Type> {
        self.imp.type_of_expr(expr)
    }

    pub fn type_of_pat(&self, pat: &ast::Pat) -> Option<Type> {
        self.imp.type_of_pat(pat)
    }

    pub fn type_of_self(&self, param: &ast::SelfParam) -> Option<Type> {
        self.imp.type_of_self(param)
    }

    pub fn resolve_method_call(&self, call: &ast::MethodCallExpr) -> Option<Function> {
        self.imp.resolve_method_call(call).map(Function::from)
    }

    pub fn resolve_method_call_as_callable(&self, call: &ast::MethodCallExpr) -> Option<Callable> {
        self.imp.resolve_method_call_as_callable(call)
    }

    pub fn resolve_field(&self, field: &ast::FieldExpr) -> Option<Field> {
        self.imp.resolve_field(field)
    }

    pub fn resolve_record_field(
        &self,
        field: &ast::RecordExprField,
    ) -> Option<(Field, Option<Local>)> {
        self.imp.resolve_record_field(field)
    }

    pub fn resolve_record_pat_field(&self, field: &ast::RecordPatField) -> Option<Field> {
        self.imp.resolve_record_pat_field(field)
    }

    pub fn resolve_macro_call(&self, macro_call: &ast::MacroCall) -> Option<MacroDef> {
        self.imp.resolve_macro_call(macro_call)
    }

    pub fn resolve_path(&self, path: &ast::Path) -> Option<PathResolution> {
        self.imp.resolve_path(path)
    }

    pub fn resolve_extern_crate(&self, extern_crate: &ast::ExternCrate) -> Option<Crate> {
        self.imp.resolve_extern_crate(extern_crate)
    }

    pub fn resolve_variant(&self, record_lit: ast::RecordExpr) -> Option<VariantDef> {
        self.imp.resolve_variant(record_lit).map(VariantDef::from)
    }

    pub fn resolve_bind_pat_to_const(&self, pat: &ast::IdentPat) -> Option<ModuleDef> {
        self.imp.resolve_bind_pat_to_const(pat)
    }

    // FIXME: use this instead?
    // pub fn resolve_name_ref(&self, name_ref: &ast::NameRef) -> Option<???>;

    pub fn record_literal_missing_fields(&self, literal: &ast::RecordExpr) -> Vec<(Field, Type)> {
        self.imp.record_literal_missing_fields(literal)
    }

    pub fn record_pattern_missing_fields(&self, pattern: &ast::RecordPat) -> Vec<(Field, Type)> {
        self.imp.record_pattern_missing_fields(pattern)
    }

    pub fn to_def<T: ToDef>(&self, src: &T) -> Option<T::Def> {
        let src = self.imp.find_file(src.syntax().clone()).with_value(src).cloned();
        T::to_def(&self.imp, src)
    }

    pub fn to_module_def(&self, file: FileId) -> Option<Module> {
        self.imp.to_module_def(file).next()
    }

    pub fn to_module_defs(&self, file: FileId) -> impl Iterator<Item = Module> {
        self.imp.to_module_def(file)
    }

    pub fn scope(&self, node: &SyntaxNode) -> SemanticsScope<'db> {
        self.imp.scope(node)
    }

    pub fn scope_at_offset(&self, token: &SyntaxToken, offset: TextSize) -> SemanticsScope<'db> {
        self.imp.scope_at_offset(&token.parent().unwrap(), offset)
    }

    pub fn scope_for_def(&self, def: Trait) -> SemanticsScope<'db> {
        self.imp.scope_for_def(def)
    }

    pub fn assert_contains_node(&self, node: &SyntaxNode) {
        self.imp.assert_contains_node(node)
    }

    pub fn is_unsafe_method_call(&self, method_call_expr: &ast::MethodCallExpr) -> bool {
        self.imp.is_unsafe_method_call(method_call_expr)
    }

    pub fn is_unsafe_ref_expr(&self, ref_expr: &ast::RefExpr) -> bool {
        self.imp.is_unsafe_ref_expr(ref_expr)
    }

    pub fn is_unsafe_ident_pat(&self, ident_pat: &ast::IdentPat) -> bool {
        self.imp.is_unsafe_ident_pat(ident_pat)
    }
}

impl<'db> SemanticsImpl<'db> {
    fn new(db: &'db dyn HirDatabase) -> Self {
        SemanticsImpl {
            db,
            s2d_cache: Default::default(),
            cache: Default::default(),
            expansion_info_cache: Default::default(),
        }
    }

    fn parse(&self, file_id: FileId) -> ast::SourceFile {
        let tree = self.db.parse(file_id).tree();
        self.cache(tree.syntax().clone(), file_id.into());
        tree
    }

    fn expand(&self, macro_call: &ast::MacroCall) -> Option<SyntaxNode> {
        let sa = self.analyze(macro_call.syntax());
        let file_id = sa.expand(self.db, InFile::new(sa.file_id, macro_call))?;
        let node = self.db.parse_or_expand(file_id)?;
        self.cache(node.clone(), file_id);
        Some(node)
    }

    fn speculative_expand(
        &self,
        actual_macro_call: &ast::MacroCall,
        hypothetical_args: &ast::TokenTree,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, SyntaxToken)> {
        let sa = self.analyze(actual_macro_call.syntax());
        let macro_call = InFile::new(sa.file_id, actual_macro_call);
        let krate = sa.resolver.krate()?;
        let macro_call_id = macro_call.as_call_id(self.db.upcast(), krate, |path| {
            sa.resolver.resolve_path_as_macro(self.db.upcast(), &path)
        })?;
        hir_expand::db::expand_hypothetical(
            self.db.upcast(),
            macro_call_id,
            hypothetical_args,
            token_to_map,
        )
    }

    fn descend_into_macros(&self, token: SyntaxToken) -> SyntaxToken {
        let _p = profile::span("descend_into_macros");
        let parent = match token.parent() {
            Some(it) => it,
            None => return token,
        };
        let sa = self.analyze(&parent);

        let token = successors(Some(InFile::new(sa.file_id, token)), |token| {
            self.db.check_canceled();
            let macro_call = token.value.ancestors().find_map(ast::MacroCall::cast)?;
            let tt = macro_call.token_tree()?;
            if !tt.syntax().text_range().contains_range(token.value.text_range()) {
                return None;
            }
            let file_id = sa.expand(self.db, token.with_value(&macro_call))?;
            let token = self
                .expansion_info_cache
                .borrow_mut()
                .entry(file_id)
                .or_insert_with(|| file_id.expansion_info(self.db.upcast()))
                .as_ref()?
                .map_token_down(token.as_ref())?;

            if let Some(parent) = token.value.parent() {
                self.cache(find_root(&parent), token.file_id);
            }

            Some(token)
        })
        .last()
        .unwrap();

        token.value
    }

    fn descend_node_at_offset(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> impl Iterator<Item = SyntaxNode> + '_ {
        // Handle macro token cases
        node.token_at_offset(offset)
            .map(|token| self.descend_into_macros(token))
            .map(|it| self.token_ancestors_with_macros(it))
            .flatten()
    }

    fn original_range(&self, node: &SyntaxNode) -> FileRange {
        let node = self.find_file(node.clone());
        node.as_ref().original_file_range(self.db.upcast())
    }

    fn diagnostics_display_range(&self, src: InFile<SyntaxNodePtr>) -> FileRange {
        let root = self.db.parse_or_expand(src.file_id).unwrap();
        let node = src.value.to_node(&root);
        self.cache(root, src.file_id);
        src.with_value(&node).original_file_range(self.db.upcast())
    }

    fn token_ancestors_with_macros(
        &self,
        token: SyntaxToken,
    ) -> impl Iterator<Item = SyntaxNode> + '_ {
        token.parent().into_iter().flat_map(move |parent| self.ancestors_with_macros(parent))
    }

    fn ancestors_with_macros(&self, node: SyntaxNode) -> impl Iterator<Item = SyntaxNode> + '_ {
        let node = self.find_file(node);
        node.ancestors_with_macros(self.db.upcast()).map(|it| it.value)
    }

    fn ancestors_at_offset_with_macros(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> impl Iterator<Item = SyntaxNode> + '_ {
        node.token_at_offset(offset)
            .map(|token| self.token_ancestors_with_macros(token))
            .kmerge_by(|node1, node2| node1.text_range().len() < node2.text_range().len())
    }

    fn resolve_lifetime_param(&self, lifetime: &ast::Lifetime) -> Option<LifetimeParam> {
        let text = lifetime.text();
        let lifetime_param = lifetime.syntax().ancestors().find_map(|syn| {
            let gpl = match_ast! {
                match syn {
                    ast::Fn(it) => it.generic_param_list()?,
                    ast::TypeAlias(it) => it.generic_param_list()?,
                    ast::Struct(it) => it.generic_param_list()?,
                    ast::Enum(it) => it.generic_param_list()?,
                    ast::Union(it) => it.generic_param_list()?,
                    ast::Trait(it) => it.generic_param_list()?,
                    ast::Impl(it) => it.generic_param_list()?,
                    ast::WherePred(it) => it.generic_param_list()?,
                    ast::ForType(it) => it.generic_param_list()?,
                    _ => return None,
                }
            };
            gpl.lifetime_params()
                .find(|tp| tp.lifetime().as_ref().map(|lt| lt.text()) == Some(text))
        })?;
        let src = self.find_file(lifetime_param.syntax().clone()).with_value(lifetime_param);
        ToDef::to_def(self, src)
    }

    fn resolve_label(&self, lifetime: &ast::Lifetime) -> Option<Label> {
        let text = lifetime.text();
        let label = lifetime.syntax().ancestors().find_map(|syn| {
            let label = match_ast! {
                match syn {
                    ast::ForExpr(it) => it.label(),
                    ast::WhileExpr(it) => it.label(),
                    ast::LoopExpr(it) => it.label(),
                    ast::EffectExpr(it) => it.label(),
                    _ => None,
                }
            };
            label.filter(|l| {
                l.lifetime()
                    .and_then(|lt| lt.lifetime_ident_token())
                    .map_or(false, |lt| lt.text() == text)
            })
        })?;
        let src = self.find_file(label.syntax().clone()).with_value(label);
        ToDef::to_def(self, src)
    }

    fn type_of_expr(&self, expr: &ast::Expr) -> Option<Type> {
        self.analyze(expr.syntax()).type_of_expr(self.db, expr)
    }

    fn type_of_pat(&self, pat: &ast::Pat) -> Option<Type> {
        self.analyze(pat.syntax()).type_of_pat(self.db, pat)
    }

    fn type_of_self(&self, param: &ast::SelfParam) -> Option<Type> {
        self.analyze(param.syntax()).type_of_self(self.db, param)
    }

    fn resolve_method_call(&self, call: &ast::MethodCallExpr) -> Option<FunctionId> {
        self.analyze(call.syntax()).resolve_method_call(self.db, call)
    }

    fn resolve_method_call_as_callable(&self, call: &ast::MethodCallExpr) -> Option<Callable> {
        // FIXME: this erases Substs
        let func = self.resolve_method_call(call)?;
        let ty = self.db.value_ty(func.into());
        let resolver = self.analyze(call.syntax()).resolver;
        let ty = Type::new_with_resolver(self.db, &resolver, ty.value)?;
        let mut res = ty.as_callable(self.db)?;
        res.is_bound_method = true;
        Some(res)
    }

    fn resolve_field(&self, field: &ast::FieldExpr) -> Option<Field> {
        self.analyze(field.syntax()).resolve_field(self.db, field)
    }

    fn resolve_record_field(&self, field: &ast::RecordExprField) -> Option<(Field, Option<Local>)> {
        self.analyze(field.syntax()).resolve_record_field(self.db, field)
    }

    fn resolve_record_pat_field(&self, field: &ast::RecordPatField) -> Option<Field> {
        self.analyze(field.syntax()).resolve_record_pat_field(self.db, field)
    }

    fn resolve_macro_call(&self, macro_call: &ast::MacroCall) -> Option<MacroDef> {
        let sa = self.analyze(macro_call.syntax());
        let macro_call = self.find_file(macro_call.syntax().clone()).with_value(macro_call);
        sa.resolve_macro_call(self.db, macro_call)
    }

    fn resolve_path(&self, path: &ast::Path) -> Option<PathResolution> {
        self.analyze(path.syntax()).resolve_path(self.db, path)
    }

    fn resolve_extern_crate(&self, extern_crate: &ast::ExternCrate) -> Option<Crate> {
        let krate = self.scope(extern_crate.syntax()).krate()?;
        krate.dependencies(self.db).into_iter().find_map(|dep| {
            if dep.name == extern_crate.name_ref()?.as_name() {
                Some(dep.krate)
            } else {
                None
            }
        })
    }

    fn resolve_variant(&self, record_lit: ast::RecordExpr) -> Option<VariantId> {
        self.analyze(record_lit.syntax()).resolve_variant(self.db, record_lit)
    }

    fn resolve_bind_pat_to_const(&self, pat: &ast::IdentPat) -> Option<ModuleDef> {
        self.analyze(pat.syntax()).resolve_bind_pat_to_const(self.db, pat)
    }

    fn record_literal_missing_fields(&self, literal: &ast::RecordExpr) -> Vec<(Field, Type)> {
        self.analyze(literal.syntax())
            .record_literal_missing_fields(self.db, literal)
            .unwrap_or_default()
    }

    fn record_pattern_missing_fields(&self, pattern: &ast::RecordPat) -> Vec<(Field, Type)> {
        self.analyze(pattern.syntax())
            .record_pattern_missing_fields(self.db, pattern)
            .unwrap_or_default()
    }

    fn with_ctx<F: FnOnce(&mut SourceToDefCtx) -> T, T>(&self, f: F) -> T {
        let mut cache = self.s2d_cache.borrow_mut();
        let mut ctx = SourceToDefCtx { db: self.db, cache: &mut *cache };
        f(&mut ctx)
    }

    fn to_module_def(&self, file: FileId) -> impl Iterator<Item = Module> {
        self.with_ctx(|ctx| ctx.file_to_def(file)).into_iter().map(Module::from)
    }

    fn scope(&self, node: &SyntaxNode) -> SemanticsScope<'db> {
        let sa = self.analyze(node);
        SemanticsScope { db: self.db, file_id: sa.file_id, resolver: sa.resolver }
    }

    fn scope_at_offset(&self, node: &SyntaxNode, offset: TextSize) -> SemanticsScope<'db> {
        let sa = self.analyze_with_offset(node, offset);
        SemanticsScope { db: self.db, file_id: sa.file_id, resolver: sa.resolver }
    }

    fn scope_for_def(&self, def: Trait) -> SemanticsScope<'db> {
        let file_id = self.db.lookup_intern_trait(def.id).id.file_id;
        let resolver = def.id.resolver(self.db.upcast());
        SemanticsScope { db: self.db, file_id, resolver }
    }

    fn analyze(&self, node: &SyntaxNode) -> SourceAnalyzer {
        self.analyze_impl(node, None)
    }
    fn analyze_with_offset(&self, node: &SyntaxNode, offset: TextSize) -> SourceAnalyzer {
        self.analyze_impl(node, Some(offset))
    }
    fn analyze_impl(&self, node: &SyntaxNode, offset: Option<TextSize>) -> SourceAnalyzer {
        let _p = profile::span("Semantics::analyze_impl");
        let node = self.find_file(node.clone());
        let node = node.as_ref();

        let container = match self.with_ctx(|ctx| ctx.find_container(node)) {
            Some(it) => it,
            None => return SourceAnalyzer::new_for_resolver(Resolver::default(), node),
        };

        let resolver = match container {
            ChildContainer::DefWithBodyId(def) => {
                return SourceAnalyzer::new_for_body(self.db, def, node, offset)
            }
            ChildContainer::TraitId(it) => it.resolver(self.db.upcast()),
            ChildContainer::ImplId(it) => it.resolver(self.db.upcast()),
            ChildContainer::ModuleId(it) => it.resolver(self.db.upcast()),
            ChildContainer::EnumId(it) => it.resolver(self.db.upcast()),
            ChildContainer::VariantId(it) => it.resolver(self.db.upcast()),
            ChildContainer::TypeAliasId(it) => it.resolver(self.db.upcast()),
            ChildContainer::GenericDefId(it) => it.resolver(self.db.upcast()),
        };
        SourceAnalyzer::new_for_resolver(resolver, node)
    }

    fn cache(&self, root_node: SyntaxNode, file_id: HirFileId) {
        assert!(root_node.parent().is_none());
        let mut cache = self.cache.borrow_mut();
        let prev = cache.insert(root_node, file_id);
        assert!(prev == None || prev == Some(file_id))
    }

    fn assert_contains_node(&self, node: &SyntaxNode) {
        self.find_file(node.clone());
    }

    fn lookup(&self, root_node: &SyntaxNode) -> Option<HirFileId> {
        let cache = self.cache.borrow();
        cache.get(root_node).copied()
    }

    fn find_file(&self, node: SyntaxNode) -> InFile<SyntaxNode> {
        let root_node = find_root(&node);
        let file_id = self.lookup(&root_node).unwrap_or_else(|| {
            panic!(
                "\n\nFailed to lookup {:?} in this Semantics.\n\
                 Make sure to use only query nodes, derived from this instance of Semantics.\n\
                 root node:   {:?}\n\
                 known nodes: {}\n\n",
                node,
                root_node,
                self.cache
                    .borrow()
                    .keys()
                    .map(|it| format!("{:?}", it))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        });
        InFile::new(file_id, node)
    }

    fn is_unsafe_method_call(&self, method_call_expr: &ast::MethodCallExpr) -> bool {
        method_call_expr
            .receiver()
            .and_then(|expr| {
                let field_expr = match expr {
                    ast::Expr::FieldExpr(field_expr) => field_expr,
                    _ => return None,
                };
                let ty = self.type_of_expr(&field_expr.expr()?)?;
                if !ty.is_packed(self.db) {
                    return None;
                }

                let func = self.resolve_method_call(&method_call_expr).map(Function::from)?;
                let res = match func.self_param(self.db)?.access(self.db) {
                    Access::Shared | Access::Exclusive => true,
                    Access::Owned => false,
                };
                Some(res)
            })
            .unwrap_or(false)
    }

    fn is_unsafe_ref_expr(&self, ref_expr: &ast::RefExpr) -> bool {
        ref_expr
            .expr()
            .and_then(|expr| {
                let field_expr = match expr {
                    ast::Expr::FieldExpr(field_expr) => field_expr,
                    _ => return None,
                };
                let expr = field_expr.expr()?;
                self.type_of_expr(&expr)
            })
            // Binding a reference to a packed type is possibly unsafe.
            .map(|ty| ty.is_packed(self.db))
            .unwrap_or(false)

        // FIXME This needs layout computation to be correct. It will highlight
        // more than it should with the current implementation.
    }

    fn is_unsafe_ident_pat(&self, ident_pat: &ast::IdentPat) -> bool {
        if !ident_pat.ref_token().is_some() {
            return false;
        }

        ident_pat
            .syntax()
            .parent()
            .and_then(|parent| {
                // `IdentPat` can live under `RecordPat` directly under `RecordPatField` or
                // `RecordPatFieldList`. `RecordPatField` also lives under `RecordPatFieldList`,
                // so this tries to lookup the `IdentPat` anywhere along that structure to the
                // `RecordPat` so we can get the containing type.
                let record_pat = ast::RecordPatField::cast(parent.clone())
                    .and_then(|record_pat| record_pat.syntax().parent())
                    .or_else(|| Some(parent.clone()))
                    .and_then(|parent| {
                        ast::RecordPatFieldList::cast(parent)?
                            .syntax()
                            .parent()
                            .and_then(ast::RecordPat::cast)
                    });

                // If this doesn't match a `RecordPat`, fallback to a `LetStmt` to see if
                // this is initialized from a `FieldExpr`.
                if let Some(record_pat) = record_pat {
                    self.type_of_pat(&ast::Pat::RecordPat(record_pat))
                } else if let Some(let_stmt) = ast::LetStmt::cast(parent) {
                    let field_expr = match let_stmt.initializer()? {
                        ast::Expr::FieldExpr(field_expr) => field_expr,
                        _ => return None,
                    };

                    self.type_of_expr(&field_expr.expr()?)
                } else {
                    None
                }
            })
            // Binding a reference to a packed type is possibly unsafe.
            .map(|ty| ty.is_packed(self.db))
            .unwrap_or(false)
    }
}

pub trait ToDef: AstNode + Clone {
    type Def;

    fn to_def(sema: &SemanticsImpl, src: InFile<Self>) -> Option<Self::Def>;
}

macro_rules! to_def_impls {
    ($(($def:path, $ast:path, $meth:ident)),* ,) => {$(
        impl ToDef for $ast {
            type Def = $def;
            fn to_def(sema: &SemanticsImpl, src: InFile<Self>) -> Option<Self::Def> {
                sema.with_ctx(|ctx| ctx.$meth(src)).map(<$def>::from)
            }
        }
    )*}
}

to_def_impls![
    (crate::Module, ast::Module, module_to_def),
    (crate::Module, ast::SourceFile, source_file_to_def),
    (crate::Struct, ast::Struct, struct_to_def),
    (crate::Enum, ast::Enum, enum_to_def),
    (crate::Union, ast::Union, union_to_def),
    (crate::Trait, ast::Trait, trait_to_def),
    (crate::Impl, ast::Impl, impl_to_def),
    (crate::TypeAlias, ast::TypeAlias, type_alias_to_def),
    (crate::Const, ast::Const, const_to_def),
    (crate::Static, ast::Static, static_to_def),
    (crate::Function, ast::Fn, fn_to_def),
    (crate::Field, ast::RecordField, record_field_to_def),
    (crate::Field, ast::TupleField, tuple_field_to_def),
    (crate::Variant, ast::Variant, enum_variant_to_def),
    (crate::TypeParam, ast::TypeParam, type_param_to_def),
    (crate::LifetimeParam, ast::LifetimeParam, lifetime_param_to_def),
    (crate::ConstParam, ast::ConstParam, const_param_to_def),
    (crate::MacroDef, ast::MacroRules, macro_rules_to_def),
    (crate::Local, ast::IdentPat, bind_pat_to_def),
    (crate::Local, ast::SelfParam, self_param_to_def),
    (crate::Label, ast::Label, label_to_def),
];

fn find_root(node: &SyntaxNode) -> SyntaxNode {
    node.ancestors().last().unwrap()
}

/// `SemanticScope` encapsulates the notion of a scope (the set of visible
/// names) at a particular program point.
///
/// It is a bit tricky, as scopes do not really exist inside the compiler.
/// Rather, the compiler directly computes for each reference the definition it
/// refers to. It might transiently compute the explicit scope map while doing
/// so, but, generally, this is not something left after the analysis.
///
/// However, we do very much need explicit scopes for IDE purposes --
/// completion, at its core, lists the contents of the current scope. The notion
/// of scope is also useful to answer questions like "what would be the meaning
/// of this piece of code if we inserted it into this position?".
///
/// So `SemanticsScope` is constructed from a specific program point (a syntax
/// node or just a raw offset) and provides access to the set of visible names
/// on a somewhat best-effort basis.
///
/// Note that if you are wondering "what does this specific existing name mean?",
/// you'd better use the `resolve_` family of methods.
#[derive(Debug)]
pub struct SemanticsScope<'a> {
    pub db: &'a dyn HirDatabase,
    file_id: HirFileId,
    resolver: Resolver,
}

impl<'a> SemanticsScope<'a> {
    pub fn module(&self) -> Option<Module> {
        Some(Module { id: self.resolver.module()? })
    }

    pub fn krate(&self) -> Option<Crate> {
        Some(Crate { id: self.resolver.krate()? })
    }

    /// Note: `FxHashSet<TraitId>` should be treated as an opaque type, passed into `Type
    // FIXME: rename to visible_traits to not repeat scope?
    pub fn traits_in_scope(&self) -> FxHashSet<TraitId> {
        let resolver = &self.resolver;
        resolver.traits_in_scope(self.db.upcast())
    }

    pub fn process_all_names(&self, f: &mut dyn FnMut(Name, ScopeDef)) {
        let resolver = &self.resolver;

        resolver.process_all_names(self.db.upcast(), &mut |name, def| {
            let def = match def {
                resolver::ScopeDef::PerNs(it) => {
                    let items = ScopeDef::all_items(it);
                    for item in items {
                        f(name.clone(), item);
                    }
                    return;
                }
                resolver::ScopeDef::ImplSelfType(it) => ScopeDef::ImplSelfType(it.into()),
                resolver::ScopeDef::AdtSelfType(it) => ScopeDef::AdtSelfType(it.into()),
                resolver::ScopeDef::GenericParam(id) => ScopeDef::GenericParam(id.into()),
                resolver::ScopeDef::Local(pat_id) => {
                    let parent = resolver.body_owner().unwrap();
                    ScopeDef::Local(Local { parent, pat_id })
                }
            };
            f(name, def)
        })
    }

    /// Resolve a path as-if it was written at the given scope. This is
    /// necessary a heuristic, as it doesn't take hygiene into account.
    pub fn speculative_resolve(&self, path: &ast::Path) -> Option<PathResolution> {
        let hygiene = Hygiene::new(self.db.upcast(), self.file_id);
        let path = Path::from_src(path.clone(), &hygiene)?;
        resolve_hir_path(self.db, &self.resolver, &path)
    }
}
