//! See `Semantics`.

mod source_to_def;

use std::{cell::RefCell, fmt, iter, mem, ops};

use base_db::{FileId, FileRange};
use either::Either;
use hir_def::{
    body,
    expr::Expr,
    macro_id_to_def_id,
    resolver::{self, HasResolver, Resolver, TypeNs},
    type_ref::Mutability,
    AsMacroCall, DefWithBodyId, FieldId, FunctionId, MacroId, TraitId, VariantId,
};
use hir_expand::{
    db::ExpandDatabase,
    name::{known, AsName},
    ExpansionInfo, MacroCallId,
};
use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{smallvec, SmallVec};
use syntax::{
    algo::skip_trivia_token,
    ast::{self, HasAttrs as _, HasGenericParams, HasLoopBody},
    match_ast, AstNode, Direction, SyntaxKind, SyntaxNode, SyntaxNodePtr, SyntaxToken, TextSize,
};

use crate::{
    db::HirDatabase,
    semantics::source_to_def::{ChildContainer, SourceToDefCache, SourceToDefCtx},
    source_analyzer::{resolve_hir_path, SourceAnalyzer},
    Access, Adjust, Adjustment, AutoBorrow, BindingMode, BuiltinAttr, Callable, ConstParam, Crate,
    DeriveHelper, Field, Function, HasSource, HirFileId, Impl, InFile, Label, LifetimeParam, Local,
    Macro, Module, ModuleDef, Name, OverloadedDeref, Path, ScopeDef, ToolModule, Trait, Type,
    TypeAlias, TypeParam, VariantDef,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathResolution {
    /// An item
    Def(ModuleDef),
    /// A local binding (only value namespace)
    Local(Local),
    /// A type parameter
    TypeParam(TypeParam),
    /// A const parameter
    ConstParam(ConstParam),
    SelfType(Impl),
    BuiltinAttr(BuiltinAttr),
    ToolModule(ToolModule),
    DeriveHelper(DeriveHelper),
}

impl PathResolution {
    pub(crate) fn in_type_ns(&self) -> Option<TypeNs> {
        match self {
            PathResolution::Def(ModuleDef::Adt(adt)) => Some(TypeNs::AdtId((*adt).into())),
            PathResolution::Def(ModuleDef::BuiltinType(builtin)) => {
                Some(TypeNs::BuiltinType((*builtin).into()))
            }
            PathResolution::Def(
                ModuleDef::Const(_)
                | ModuleDef::Variant(_)
                | ModuleDef::Macro(_)
                | ModuleDef::Function(_)
                | ModuleDef::Module(_)
                | ModuleDef::Static(_)
                | ModuleDef::Trait(_)
                | ModuleDef::TraitAlias(_),
            ) => None,
            PathResolution::Def(ModuleDef::TypeAlias(alias)) => {
                Some(TypeNs::TypeAliasId((*alias).into()))
            }
            PathResolution::BuiltinAttr(_)
            | PathResolution::ToolModule(_)
            | PathResolution::Local(_)
            | PathResolution::DeriveHelper(_)
            | PathResolution::ConstParam(_) => None,
            PathResolution::TypeParam(param) => Some(TypeNs::GenericParam((*param).into())),
            PathResolution::SelfType(impl_def) => Some(TypeNs::SelfType((*impl_def).into())),
        }
    }
}

#[derive(Debug)]
pub struct TypeInfo {
    /// The original type of the expression or pattern.
    pub original: Type,
    /// The adjusted type, if an adjustment happened.
    pub adjusted: Option<Type>,
}

impl TypeInfo {
    pub fn original(self) -> Type {
        self.original
    }

    pub fn has_adjustment(&self) -> bool {
        self.adjusted.is_some()
    }

    /// The adjusted type, or the original in case no adjustments occurred.
    pub fn adjusted(self) -> Type {
        self.adjusted.unwrap_or(self.original)
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
    // Rootnode to HirFileId cache
    cache: RefCell<FxHashMap<SyntaxNode, HirFileId>>,
    // MacroCall to its expansion's HirFileId cache
    macro_call_cache: RefCell<FxHashMap<InFile<ast::MacroCall>, HirFileId>>,
}

impl<DB> fmt::Debug for Semantics<'_, DB> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Semantics {{ ... }}")
    }
}

impl<'db, DB: HirDatabase> Semantics<'db, DB> {
    pub fn new(db: &DB) -> Semantics<'_, DB> {
        let impl_ = SemanticsImpl::new(db);
        Semantics { db, imp: impl_ }
    }

    pub fn parse(&self, file_id: FileId) -> ast::SourceFile {
        self.imp.parse(file_id)
    }

    pub fn parse_or_expand(&self, file_id: HirFileId) -> Option<SyntaxNode> {
        self.imp.parse_or_expand(file_id)
    }

    pub fn expand(&self, macro_call: &ast::MacroCall) -> Option<SyntaxNode> {
        self.imp.expand(macro_call)
    }

    /// If `item` has an attribute macro attached to it, expands it.
    pub fn expand_attr_macro(&self, item: &ast::Item) -> Option<SyntaxNode> {
        self.imp.expand_attr_macro(item)
    }

    pub fn expand_derive_as_pseudo_attr_macro(&self, attr: &ast::Attr) -> Option<SyntaxNode> {
        self.imp.expand_derive_as_pseudo_attr_macro(attr)
    }

    pub fn resolve_derive_macro(&self, derive: &ast::Attr) -> Option<Vec<Option<Macro>>> {
        self.imp.resolve_derive_macro(derive)
    }

    pub fn expand_derive_macro(&self, derive: &ast::Attr) -> Option<Vec<SyntaxNode>> {
        self.imp.expand_derive_macro(derive)
    }

    pub fn is_attr_macro_call(&self, item: &ast::Item) -> bool {
        self.imp.is_attr_macro_call(item)
    }

    pub fn is_derive_annotated(&self, item: &ast::Adt) -> bool {
        self.imp.is_derive_annotated(item)
    }

    pub fn speculative_expand(
        &self,
        actual_macro_call: &ast::MacroCall,
        speculative_args: &ast::TokenTree,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, SyntaxToken)> {
        self.imp.speculative_expand(actual_macro_call, speculative_args, token_to_map)
    }

    pub fn speculative_expand_attr_macro(
        &self,
        actual_macro_call: &ast::Item,
        speculative_args: &ast::Item,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, SyntaxToken)> {
        self.imp.speculative_expand_attr(actual_macro_call, speculative_args, token_to_map)
    }

    pub fn speculative_expand_derive_as_pseudo_attr_macro(
        &self,
        actual_macro_call: &ast::Attr,
        speculative_args: &ast::Attr,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, SyntaxToken)> {
        self.imp.speculative_expand_derive_as_pseudo_attr_macro(
            actual_macro_call,
            speculative_args,
            token_to_map,
        )
    }

    /// Descend the token into macrocalls to its first mapped counterpart.
    pub fn descend_into_macros_single(&self, token: SyntaxToken) -> SyntaxToken {
        self.imp.descend_into_macros_single(token)
    }

    /// Descend the token into macrocalls to all its mapped counterparts.
    pub fn descend_into_macros(&self, token: SyntaxToken) -> SmallVec<[SyntaxToken; 1]> {
        self.imp.descend_into_macros(token)
    }

    /// Descend the token into macrocalls to all its mapped counterparts that have the same text as the input token.
    ///
    /// Returns the original non descended token if none of the mapped counterparts have the same text.
    pub fn descend_into_macros_with_same_text(
        &self,
        token: SyntaxToken,
    ) -> SmallVec<[SyntaxToken; 1]> {
        self.imp.descend_into_macros_with_same_text(token)
    }

    pub fn descend_into_macros_with_kind_preference(&self, token: SyntaxToken) -> SyntaxToken {
        self.imp.descend_into_macros_with_kind_preference(token)
    }

    /// Maps a node down by mapping its first and last token down.
    pub fn descend_node_into_attributes<N: AstNode>(&self, node: N) -> SmallVec<[N; 1]> {
        self.imp.descend_node_into_attributes(node)
    }

    /// Search for a definition's source and cache its syntax tree
    pub fn source<Def: HasSource>(&self, def: Def) -> Option<InFile<Def::Ast>>
    where
        Def::Ast: AstNode,
    {
        self.imp.source(def)
    }

    pub fn hir_file_for(&self, syntax_node: &SyntaxNode) -> HirFileId {
        self.imp.find_file(syntax_node).file_id
    }

    /// Attempts to map the node out of macro expanded files returning the original file range.
    /// If upmapping is not possible, this will fall back to the range of the macro call of the
    /// macro file the node resides in.
    pub fn original_range(&self, node: &SyntaxNode) -> FileRange {
        self.imp.original_range(node)
    }

    /// Attempts to map the node out of macro expanded files returning the original file range.
    pub fn original_range_opt(&self, node: &SyntaxNode) -> Option<FileRange> {
        self.imp.original_range_opt(node)
    }

    /// Attempts to map the node out of macro expanded files.
    /// This only work for attribute expansions, as other ones do not have nodes as input.
    pub fn original_ast_node<N: AstNode>(&self, node: N) -> Option<N> {
        self.imp.original_ast_node(node)
    }
    /// Attempts to map the node out of macro expanded files.
    /// This only work for attribute expansions, as other ones do not have nodes as input.
    pub fn original_syntax_node(&self, node: &SyntaxNode) -> Option<SyntaxNode> {
        self.imp.original_syntax_node(node)
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

    /// Iterates the ancestors of the given node, climbing up macro expansions while doing so.
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

    /// Find an AstNode by offset inside SyntaxNode, if it is inside *Macrofile*,
    /// search up until it is of the target AstNode type
    pub fn find_node_at_offset_with_macros<N: AstNode>(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> Option<N> {
        self.imp.ancestors_at_offset_with_macros(node, offset).find_map(N::cast)
    }

    /// Find an AstNode by offset inside SyntaxNode, if it is inside *MacroCall*,
    /// descend it and find again
    pub fn find_node_at_offset_with_descend<N: AstNode>(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> Option<N> {
        self.imp.descend_node_at_offset(node, offset).flatten().find_map(N::cast)
    }

    /// Find an AstNode by offset inside SyntaxNode, if it is inside *MacroCall*,
    /// descend it and find again
    pub fn find_nodes_at_offset_with_descend<'slf, N: AstNode + 'slf>(
        &'slf self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> impl Iterator<Item = N> + 'slf {
        self.imp.descend_node_at_offset(node, offset).filter_map(|mut it| it.find_map(N::cast))
    }

    pub fn resolve_lifetime_param(&self, lifetime: &ast::Lifetime) -> Option<LifetimeParam> {
        self.imp.resolve_lifetime_param(lifetime)
    }

    pub fn resolve_label(&self, lifetime: &ast::Lifetime) -> Option<Label> {
        self.imp.resolve_label(lifetime)
    }

    pub fn resolve_type(&self, ty: &ast::Type) -> Option<Type> {
        self.imp.resolve_type(ty)
    }

    pub fn resolve_trait(&self, trait_: &ast::Path) -> Option<Trait> {
        self.imp.resolve_trait(trait_)
    }

    pub fn expr_adjustments(&self, expr: &ast::Expr) -> Option<Vec<Adjustment>> {
        self.imp.expr_adjustments(expr)
    }

    pub fn type_of_expr(&self, expr: &ast::Expr) -> Option<TypeInfo> {
        self.imp.type_of_expr(expr)
    }

    pub fn type_of_pat(&self, pat: &ast::Pat) -> Option<TypeInfo> {
        self.imp.type_of_pat(pat)
    }

    pub fn type_of_self(&self, param: &ast::SelfParam) -> Option<Type> {
        self.imp.type_of_self(param)
    }

    pub fn pattern_adjustments(&self, pat: &ast::Pat) -> SmallVec<[Type; 1]> {
        self.imp.pattern_adjustments(pat)
    }

    pub fn binding_mode_of_pat(&self, pat: &ast::IdentPat) -> Option<BindingMode> {
        self.imp.binding_mode_of_pat(pat)
    }

    pub fn resolve_method_call(&self, call: &ast::MethodCallExpr) -> Option<Function> {
        self.imp.resolve_method_call(call).map(Function::from)
    }

    /// Attempts to resolve this call expression as a method call falling back to resolving it as a field.
    pub fn resolve_method_call_field_fallback(
        &self,
        call: &ast::MethodCallExpr,
    ) -> Option<Either<Function, Field>> {
        self.imp
            .resolve_method_call_fallback(call)
            .map(|it| it.map_left(Function::from).map_right(Field::from))
    }

    pub fn resolve_await_to_poll(&self, await_expr: &ast::AwaitExpr) -> Option<Function> {
        self.imp.resolve_await_to_poll(await_expr).map(Function::from)
    }

    pub fn resolve_prefix_expr(&self, prefix_expr: &ast::PrefixExpr) -> Option<Function> {
        self.imp.resolve_prefix_expr(prefix_expr).map(Function::from)
    }

    pub fn resolve_index_expr(&self, index_expr: &ast::IndexExpr) -> Option<Function> {
        self.imp.resolve_index_expr(index_expr).map(Function::from)
    }

    pub fn resolve_bin_expr(&self, bin_expr: &ast::BinExpr) -> Option<Function> {
        self.imp.resolve_bin_expr(bin_expr).map(Function::from)
    }

    pub fn resolve_try_expr(&self, try_expr: &ast::TryExpr) -> Option<Function> {
        self.imp.resolve_try_expr(try_expr).map(Function::from)
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
    ) -> Option<(Field, Option<Local>, Type)> {
        self.imp.resolve_record_field(field)
    }

    pub fn resolve_record_pat_field(&self, field: &ast::RecordPatField) -> Option<(Field, Type)> {
        self.imp.resolve_record_pat_field(field)
    }

    pub fn resolve_macro_call(&self, macro_call: &ast::MacroCall) -> Option<Macro> {
        self.imp.resolve_macro_call(macro_call)
    }

    pub fn is_unsafe_macro_call(&self, macro_call: &ast::MacroCall) -> bool {
        self.imp.is_unsafe_macro_call(macro_call)
    }

    pub fn resolve_attr_macro_call(&self, item: &ast::Item) -> Option<Macro> {
        self.imp.resolve_attr_macro_call(item)
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

    pub fn record_literal_missing_fields(&self, literal: &ast::RecordExpr) -> Vec<(Field, Type)> {
        self.imp.record_literal_missing_fields(literal)
    }

    pub fn record_pattern_missing_fields(&self, pattern: &ast::RecordPat) -> Vec<(Field, Type)> {
        self.imp.record_pattern_missing_fields(pattern)
    }

    pub fn to_def<T: ToDef>(&self, src: &T) -> Option<T::Def> {
        self.imp.to_def(src)
    }

    pub fn to_module_def(&self, file: FileId) -> Option<Module> {
        self.imp.to_module_def(file).next()
    }

    pub fn to_module_defs(&self, file: FileId) -> impl Iterator<Item = Module> {
        self.imp.to_module_def(file)
    }

    pub fn scope(&self, node: &SyntaxNode) -> Option<SemanticsScope<'db>> {
        self.imp.scope(node)
    }

    pub fn scope_at_offset(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> Option<SemanticsScope<'db>> {
        self.imp.scope_at_offset(node, offset)
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

    /// Returns `true` if the `node` is inside an `unsafe` context.
    pub fn is_inside_unsafe(&self, expr: &ast::Expr) -> bool {
        self.imp.is_inside_unsafe(expr)
    }
}

impl<'db> SemanticsImpl<'db> {
    fn new(db: &'db dyn HirDatabase) -> Self {
        SemanticsImpl {
            db,
            s2d_cache: Default::default(),
            cache: Default::default(),
            expansion_info_cache: Default::default(),
            macro_call_cache: Default::default(),
        }
    }

    fn parse(&self, file_id: FileId) -> ast::SourceFile {
        let tree = self.db.parse(file_id).tree();
        self.cache(tree.syntax().clone(), file_id.into());
        tree
    }

    fn parse_or_expand(&self, file_id: HirFileId) -> Option<SyntaxNode> {
        let node = self.db.parse_or_expand(file_id)?;
        self.cache(node.clone(), file_id);
        Some(node)
    }

    fn expand(&self, macro_call: &ast::MacroCall) -> Option<SyntaxNode> {
        let sa = self.analyze_no_infer(macro_call.syntax())?;
        let file_id = sa.expand(self.db, InFile::new(sa.file_id, macro_call))?;
        let node = self.parse_or_expand(file_id)?;
        Some(node)
    }

    fn expand_attr_macro(&self, item: &ast::Item) -> Option<SyntaxNode> {
        let src = self.wrap_node_infile(item.clone());
        let macro_call_id = self.with_ctx(|ctx| ctx.item_to_macro_call(src))?;
        self.parse_or_expand(macro_call_id.as_file())
    }

    fn expand_derive_as_pseudo_attr_macro(&self, attr: &ast::Attr) -> Option<SyntaxNode> {
        let adt = attr.syntax().parent().and_then(ast::Adt::cast)?;
        let src = self.wrap_node_infile(attr.clone());
        let call_id = self.with_ctx(|ctx| {
            ctx.attr_to_derive_macro_call(src.with_value(&adt), src).map(|(_, it, _)| it)
        })?;
        self.parse_or_expand(call_id.as_file())
    }

    fn resolve_derive_macro(&self, attr: &ast::Attr) -> Option<Vec<Option<Macro>>> {
        let calls = self.derive_macro_calls(attr)?;
        self.with_ctx(|ctx| {
            Some(
                calls
                    .into_iter()
                    .map(|call| {
                        macro_call_to_macro_id(ctx, self.db.upcast(), call?).map(|id| Macro { id })
                    })
                    .collect(),
            )
        })
    }

    fn expand_derive_macro(&self, attr: &ast::Attr) -> Option<Vec<SyntaxNode>> {
        let res: Vec<_> = self
            .derive_macro_calls(attr)?
            .into_iter()
            .flat_map(|call| {
                let file_id = call?.as_file();
                let node = self.db.parse_or_expand(file_id)?;
                self.cache(node.clone(), file_id);
                Some(node)
            })
            .collect();
        Some(res)
    }

    fn derive_macro_calls(&self, attr: &ast::Attr) -> Option<Vec<Option<MacroCallId>>> {
        let adt = attr.syntax().parent().and_then(ast::Adt::cast)?;
        let file_id = self.find_file(adt.syntax()).file_id;
        let adt = InFile::new(file_id, &adt);
        let src = InFile::new(file_id, attr.clone());
        self.with_ctx(|ctx| {
            let (.., res) = ctx.attr_to_derive_macro_call(adt, src)?;
            Some(res.to_vec())
        })
    }

    fn is_derive_annotated(&self, adt: &ast::Adt) -> bool {
        let file_id = self.find_file(adt.syntax()).file_id;
        let adt = InFile::new(file_id, adt);
        self.with_ctx(|ctx| ctx.has_derives(adt))
    }

    fn is_attr_macro_call(&self, item: &ast::Item) -> bool {
        let file_id = self.find_file(item.syntax()).file_id;
        let src = InFile::new(file_id, item.clone());
        self.with_ctx(|ctx| ctx.item_to_macro_call(src).is_some())
    }

    fn speculative_expand(
        &self,
        actual_macro_call: &ast::MacroCall,
        speculative_args: &ast::TokenTree,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, SyntaxToken)> {
        let SourceAnalyzer { file_id, resolver, .. } =
            self.analyze_no_infer(actual_macro_call.syntax())?;
        let macro_call = InFile::new(file_id, actual_macro_call);
        let krate = resolver.krate();
        let macro_call_id = macro_call.as_call_id(self.db.upcast(), krate, |path| {
            resolver
                .resolve_path_as_macro(self.db.upcast(), &path)
                .map(|it| macro_id_to_def_id(self.db.upcast(), it))
        })?;
        hir_expand::db::expand_speculative(
            self.db.upcast(),
            macro_call_id,
            speculative_args.syntax(),
            token_to_map,
        )
    }

    fn speculative_expand_attr(
        &self,
        actual_macro_call: &ast::Item,
        speculative_args: &ast::Item,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, SyntaxToken)> {
        let macro_call = self.wrap_node_infile(actual_macro_call.clone());
        let macro_call_id = self.with_ctx(|ctx| ctx.item_to_macro_call(macro_call))?;
        hir_expand::db::expand_speculative(
            self.db.upcast(),
            macro_call_id,
            speculative_args.syntax(),
            token_to_map,
        )
    }

    fn speculative_expand_derive_as_pseudo_attr_macro(
        &self,
        actual_macro_call: &ast::Attr,
        speculative_args: &ast::Attr,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, SyntaxToken)> {
        let attr = self.wrap_node_infile(actual_macro_call.clone());
        let adt = actual_macro_call.syntax().parent().and_then(ast::Adt::cast)?;
        let macro_call_id = self.with_ctx(|ctx| {
            ctx.attr_to_derive_macro_call(attr.with_value(&adt), attr).map(|(_, it, _)| it)
        })?;
        hir_expand::db::expand_speculative(
            self.db.upcast(),
            macro_call_id,
            speculative_args.syntax(),
            token_to_map,
        )
    }

    // This might not be the correct way to do this, but it works for now
    fn descend_node_into_attributes<N: AstNode>(&self, node: N) -> SmallVec<[N; 1]> {
        let mut res = smallvec![];
        let tokens = (|| {
            let first = skip_trivia_token(node.syntax().first_token()?, Direction::Next)?;
            let last = skip_trivia_token(node.syntax().last_token()?, Direction::Prev)?;
            Some((first, last))
        })();
        let (first, last) = match tokens {
            Some(it) => it,
            None => return res,
        };

        if first == last {
            self.descend_into_macros_impl(first, &mut |InFile { value, .. }| {
                if let Some(node) = value.parent_ancestors().find_map(N::cast) {
                    res.push(node)
                }
                false
            });
        } else {
            // Descend first and last token, then zip them to look for the node they belong to
            let mut scratch: SmallVec<[_; 1]> = smallvec![];
            self.descend_into_macros_impl(first, &mut |token| {
                scratch.push(token);
                false
            });

            let mut scratch = scratch.into_iter();
            self.descend_into_macros_impl(
                last,
                &mut |InFile { value: last, file_id: last_fid }| {
                    if let Some(InFile { value: first, file_id: first_fid }) = scratch.next() {
                        if first_fid == last_fid {
                            if let Some(p) = first.parent() {
                                let range = first.text_range().cover(last.text_range());
                                let node = find_root(&p)
                                    .covering_element(range)
                                    .ancestors()
                                    .take_while(|it| it.text_range() == range)
                                    .find_map(N::cast);
                                if let Some(node) = node {
                                    res.push(node);
                                }
                            }
                        }
                    }
                    false
                },
            );
        }
        res
    }

    fn descend_into_macros(&self, token: SyntaxToken) -> SmallVec<[SyntaxToken; 1]> {
        let mut res = smallvec![];
        self.descend_into_macros_impl(token, &mut |InFile { value, .. }| {
            res.push(value);
            false
        });
        res
    }

    fn descend_into_macros_with_same_text(&self, token: SyntaxToken) -> SmallVec<[SyntaxToken; 1]> {
        let text = token.text();
        let mut res = smallvec![];
        self.descend_into_macros_impl(token.clone(), &mut |InFile { value, .. }| {
            if value.text() == text {
                res.push(value);
            }
            false
        });
        if res.is_empty() {
            res.push(token);
        }
        res
    }

    fn descend_into_macros_with_kind_preference(&self, token: SyntaxToken) -> SyntaxToken {
        let fetch_kind = |token: &SyntaxToken| match token.parent() {
            Some(node) => match node.kind() {
                kind @ (SyntaxKind::NAME | SyntaxKind::NAME_REF) => {
                    node.parent().map_or(kind, |it| it.kind())
                }
                _ => token.kind(),
            },
            None => token.kind(),
        };
        let preferred_kind = fetch_kind(&token);
        let mut res = None;
        self.descend_into_macros_impl(token.clone(), &mut |InFile { value, .. }| {
            if fetch_kind(&value) == preferred_kind {
                res = Some(value);
                true
            } else {
                if let None = res {
                    res = Some(value)
                }
                false
            }
        });
        res.unwrap_or(token)
    }

    fn descend_into_macros_single(&self, token: SyntaxToken) -> SyntaxToken {
        let mut res = token.clone();
        self.descend_into_macros_impl(token, &mut |InFile { value, .. }| {
            res = value;
            true
        });
        res
    }

    fn descend_into_macros_impl(
        &self,
        token: SyntaxToken,
        f: &mut dyn FnMut(InFile<SyntaxToken>) -> bool,
    ) {
        let _p = profile::span("descend_into_macros");
        let parent = match token.parent() {
            Some(it) => it,
            None => return,
        };
        let sa = match self.analyze_no_infer(&parent) {
            Some(it) => it,
            None => return,
        };
        let def_map = sa.resolver.def_map();

        let mut stack: SmallVec<[_; 4]> = smallvec![InFile::new(sa.file_id, token)];
        let mut cache = self.expansion_info_cache.borrow_mut();
        let mut mcache = self.macro_call_cache.borrow_mut();

        let mut process_expansion_for_token =
            |stack: &mut SmallVec<_>, macro_file, item, token: InFile<&_>| {
                let expansion_info = cache
                    .entry(macro_file)
                    .or_insert_with(|| macro_file.expansion_info(self.db.upcast()))
                    .as_ref()?;

                {
                    let InFile { file_id, value } = expansion_info.expanded();
                    self.cache(value, file_id);
                }

                let mapped_tokens = expansion_info.map_token_down(self.db.upcast(), item, token)?;
                let len = stack.len();

                // requeue the tokens we got from mapping our current token down
                stack.extend(mapped_tokens);
                // if the length changed we have found a mapping for the token
                (stack.len() != len).then_some(())
            };

        // Remap the next token in the queue into a macro call its in, if it is not being remapped
        // either due to not being in a macro-call or because its unused push it into the result vec,
        // otherwise push the remapped tokens back into the queue as they can potentially be remapped again.
        while let Some(token) = stack.pop() {
            self.db.unwind_if_cancelled();
            let was_not_remapped = (|| {
                // First expand into attribute invocations
                let containing_attribute_macro_call = self.with_ctx(|ctx| {
                    token.value.parent_ancestors().filter_map(ast::Item::cast).find_map(|item| {
                        if item.attrs().next().is_none() {
                            // Don't force populate the dyn cache for items that don't have an attribute anyways
                            return None;
                        }
                        Some((ctx.item_to_macro_call(token.with_value(item.clone()))?, item))
                    })
                });
                if let Some((call_id, item)) = containing_attribute_macro_call {
                    let file_id = call_id.as_file();
                    return process_expansion_for_token(
                        &mut stack,
                        file_id,
                        Some(item),
                        token.as_ref(),
                    );
                }

                // Then check for token trees, that means we are either in a function-like macro or
                // secondary attribute inputs
                let tt = token.value.parent_ancestors().map_while(ast::TokenTree::cast).last()?;
                let parent = tt.syntax().parent()?;

                if tt.left_delimiter_token().map_or(false, |it| it == token.value) {
                    return None;
                }
                if tt.right_delimiter_token().map_or(false, |it| it == token.value) {
                    return None;
                }

                if let Some(macro_call) = ast::MacroCall::cast(parent.clone()) {
                    let mcall = token.with_value(macro_call);
                    let file_id = match mcache.get(&mcall) {
                        Some(&it) => it,
                        None => {
                            let it = sa.expand(self.db, mcall.as_ref())?;
                            mcache.insert(mcall, it);
                            it
                        }
                    };
                    process_expansion_for_token(&mut stack, file_id, None, token.as_ref())
                } else if let Some(meta) = ast::Meta::cast(parent) {
                    // attribute we failed expansion for earlier, this might be a derive invocation
                    // or derive helper attribute
                    let attr = meta.parent_attr()?;

                    let adt = if let Some(adt) = attr.syntax().parent().and_then(ast::Adt::cast) {
                        // this might be a derive, or a derive helper on an ADT
                        let derive_call = self.with_ctx(|ctx| {
                            // so try downmapping the token into the pseudo derive expansion
                            // see [hir_expand::builtin_attr_macro] for how the pseudo derive expansion works
                            ctx.attr_to_derive_macro_call(
                                token.with_value(&adt),
                                token.with_value(attr.clone()),
                            )
                            .map(|(_, call_id, _)| call_id)
                        });

                        match derive_call {
                            Some(call_id) => {
                                // resolved to a derive
                                let file_id = call_id.as_file();
                                return process_expansion_for_token(
                                    &mut stack,
                                    file_id,
                                    Some(adt.into()),
                                    token.as_ref(),
                                );
                            }
                            None => Some(adt),
                        }
                    } else {
                        // Otherwise this could be a derive helper on a variant or field
                        if let Some(field) = attr.syntax().parent().and_then(ast::RecordField::cast)
                        {
                            field.syntax().ancestors().take(4).find_map(ast::Adt::cast)
                        } else if let Some(field) =
                            attr.syntax().parent().and_then(ast::TupleField::cast)
                        {
                            field.syntax().ancestors().take(4).find_map(ast::Adt::cast)
                        } else if let Some(variant) =
                            attr.syntax().parent().and_then(ast::Variant::cast)
                        {
                            variant.syntax().ancestors().nth(2).and_then(ast::Adt::cast)
                        } else {
                            None
                        }
                    }?;
                    if !self.with_ctx(|ctx| ctx.has_derives(InFile::new(token.file_id, &adt))) {
                        return None;
                    }
                    // Not an attribute, nor a derive, so it's either a builtin or a derive helper
                    // Try to resolve to a derive helper and downmap
                    let attr_name = attr.path().and_then(|it| it.as_single_name_ref())?.as_name();
                    let id = self.db.ast_id_map(token.file_id).ast_id(&adt);
                    let helpers =
                        def_map.derive_helpers_in_scope(InFile::new(token.file_id, id))?;
                    let item = Some(adt.into());
                    let mut res = None;
                    for (.., derive) in helpers.iter().filter(|(helper, ..)| *helper == attr_name) {
                        res = res.or(process_expansion_for_token(
                            &mut stack,
                            derive.as_file(),
                            item.clone(),
                            token.as_ref(),
                        ));
                    }
                    res
                } else {
                    None
                }
            })()
            .is_none();

            if was_not_remapped && f(token) {
                break;
            }
        }
    }

    // Note this return type is deliberate as [`find_nodes_at_offset_with_descend`] wants to stop
    // traversing the inner iterator when it finds a node.
    // The outer iterator is over the tokens descendants
    // The inner iterator is the ancestors of a descendant
    fn descend_node_at_offset(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> impl Iterator<Item = impl Iterator<Item = SyntaxNode> + '_> + '_ {
        node.token_at_offset(offset)
            .map(move |token| self.descend_into_macros(token))
            .map(|descendants| {
                descendants.into_iter().map(move |it| self.token_ancestors_with_macros(it))
            })
            // re-order the tokens from token_at_offset by returning the ancestors with the smaller first nodes first
            // See algo::ancestors_at_offset, which uses the same approach
            .kmerge_by(|left, right| {
                left.clone()
                    .map(|node| node.text_range().len())
                    .lt(right.clone().map(|node| node.text_range().len()))
            })
    }

    fn original_range(&self, node: &SyntaxNode) -> FileRange {
        let node = self.find_file(node);
        node.original_file_range(self.db.upcast())
    }

    fn original_range_opt(&self, node: &SyntaxNode) -> Option<FileRange> {
        let node = self.find_file(node);
        node.original_file_range_opt(self.db.upcast())
    }

    fn original_ast_node<N: AstNode>(&self, node: N) -> Option<N> {
        self.wrap_node_infile(node).original_ast_node(self.db.upcast()).map(
            |InFile { file_id, value }| {
                self.cache(find_root(value.syntax()), file_id);
                value
            },
        )
    }

    fn original_syntax_node(&self, node: &SyntaxNode) -> Option<SyntaxNode> {
        let InFile { file_id, .. } = self.find_file(node);
        InFile::new(file_id, node).original_syntax_node(self.db.upcast()).map(
            |InFile { file_id, value }| {
                self.cache(find_root(&value), file_id);
                value
            },
        )
    }

    fn diagnostics_display_range(&self, src: InFile<SyntaxNodePtr>) -> FileRange {
        let root = self.parse_or_expand(src.file_id).unwrap();
        let node = src.map(|it| it.to_node(&root));
        node.as_ref().original_file_range(self.db.upcast())
    }

    fn token_ancestors_with_macros(
        &self,
        token: SyntaxToken,
    ) -> impl Iterator<Item = SyntaxNode> + Clone + '_ {
        token.parent().into_iter().flat_map(move |parent| self.ancestors_with_macros(parent))
    }

    fn ancestors_with_macros(
        &self,
        node: SyntaxNode,
    ) -> impl Iterator<Item = SyntaxNode> + Clone + '_ {
        let node = self.find_file(&node);
        let db = self.db.upcast();
        iter::successors(Some(node.cloned()), move |&InFile { file_id, ref value }| {
            match value.parent() {
                Some(parent) => Some(InFile::new(file_id, parent)),
                None => {
                    self.cache(value.clone(), file_id);
                    file_id.call_node(db)
                }
            }
        })
        .map(|it| it.value)
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
            let gpl = ast::AnyHasGenericParams::cast(syn)?.generic_param_list()?;
            gpl.lifetime_params()
                .find(|tp| tp.lifetime().as_ref().map(|lt| lt.text()).as_ref() == Some(&text))
        })?;
        let src = self.wrap_node_infile(lifetime_param);
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
                    ast::BlockExpr(it) => it.label(),
                    _ => None,
                }
            };
            label.filter(|l| {
                l.lifetime()
                    .and_then(|lt| lt.lifetime_ident_token())
                    .map_or(false, |lt| lt.text() == text)
            })
        })?;
        let src = self.wrap_node_infile(label);
        ToDef::to_def(self, src)
    }

    fn resolve_type(&self, ty: &ast::Type) -> Option<Type> {
        let analyze = self.analyze(ty.syntax())?;
        let ctx = body::LowerCtx::new(self.db.upcast(), analyze.file_id);
        let ty = hir_ty::TyLoweringContext::new(self.db, &analyze.resolver)
            .lower_ty(&crate::TypeRef::from_ast(&ctx, ty.clone()));
        Some(Type::new_with_resolver(self.db, &analyze.resolver, ty))
    }

    fn resolve_trait(&self, path: &ast::Path) -> Option<Trait> {
        let analyze = self.analyze(path.syntax())?;
        let hygiene = hir_expand::hygiene::Hygiene::new(self.db.upcast(), analyze.file_id);
        let ctx = body::LowerCtx::with_hygiene(self.db.upcast(), &hygiene);
        let hir_path = Path::from_src(path.clone(), &ctx)?;
        match analyze
            .resolver
            .resolve_path_in_type_ns_fully(self.db.upcast(), hir_path.mod_path())?
        {
            TypeNs::TraitId(id) => Some(Trait { id }),
            _ => None,
        }
    }

    fn expr_adjustments(&self, expr: &ast::Expr) -> Option<Vec<Adjustment>> {
        let mutability = |m| match m {
            hir_ty::Mutability::Not => Mutability::Shared,
            hir_ty::Mutability::Mut => Mutability::Mut,
        };

        let analyzer = self.analyze(expr.syntax())?;

        let (mut source_ty, _) = analyzer.type_of_expr(self.db, expr)?;

        analyzer.expr_adjustments(self.db, expr).map(|it| {
            it.iter()
                .map(|adjust| {
                    let target =
                        Type::new_with_resolver(self.db, &analyzer.resolver, adjust.target.clone());
                    let kind = match adjust.kind {
                        hir_ty::Adjust::NeverToAny => Adjust::NeverToAny,
                        hir_ty::Adjust::Deref(Some(hir_ty::OverloadedDeref(m))) => {
                            // FIXME: Should we handle unknown mutability better?
                            Adjust::Deref(Some(OverloadedDeref(
                                m.map(mutability).unwrap_or(Mutability::Shared),
                            )))
                        }
                        hir_ty::Adjust::Deref(None) => Adjust::Deref(None),
                        hir_ty::Adjust::Borrow(hir_ty::AutoBorrow::RawPtr(m)) => {
                            Adjust::Borrow(AutoBorrow::RawPtr(mutability(m)))
                        }
                        hir_ty::Adjust::Borrow(hir_ty::AutoBorrow::Ref(m)) => {
                            Adjust::Borrow(AutoBorrow::Ref(mutability(m)))
                        }
                        hir_ty::Adjust::Pointer(pc) => Adjust::Pointer(pc),
                    };

                    // Update `source_ty` for the next adjustment
                    let source = mem::replace(&mut source_ty, target.clone());

                    let adjustment = Adjustment { source, target, kind };

                    adjustment
                })
                .collect()
        })
    }

    fn type_of_expr(&self, expr: &ast::Expr) -> Option<TypeInfo> {
        self.analyze(expr.syntax())?
            .type_of_expr(self.db, expr)
            .map(|(ty, coerced)| TypeInfo { original: ty, adjusted: coerced })
    }

    fn type_of_pat(&self, pat: &ast::Pat) -> Option<TypeInfo> {
        self.analyze(pat.syntax())?
            .type_of_pat(self.db, pat)
            .map(|(ty, coerced)| TypeInfo { original: ty, adjusted: coerced })
    }

    fn type_of_self(&self, param: &ast::SelfParam) -> Option<Type> {
        self.analyze(param.syntax())?.type_of_self(self.db, param)
    }

    fn pattern_adjustments(&self, pat: &ast::Pat) -> SmallVec<[Type; 1]> {
        self.analyze(pat.syntax())
            .and_then(|it| it.pattern_adjustments(self.db, pat))
            .unwrap_or_default()
    }

    fn binding_mode_of_pat(&self, pat: &ast::IdentPat) -> Option<BindingMode> {
        self.analyze(pat.syntax())?.binding_mode_of_pat(self.db, pat)
    }

    fn resolve_method_call(&self, call: &ast::MethodCallExpr) -> Option<FunctionId> {
        self.analyze(call.syntax())?.resolve_method_call(self.db, call)
    }

    fn resolve_method_call_fallback(
        &self,
        call: &ast::MethodCallExpr,
    ) -> Option<Either<FunctionId, FieldId>> {
        self.analyze(call.syntax())?.resolve_method_call_fallback(self.db, call)
    }

    fn resolve_await_to_poll(&self, await_expr: &ast::AwaitExpr) -> Option<FunctionId> {
        self.analyze(await_expr.syntax())?.resolve_await_to_poll(self.db, await_expr)
    }

    fn resolve_prefix_expr(&self, prefix_expr: &ast::PrefixExpr) -> Option<FunctionId> {
        self.analyze(prefix_expr.syntax())?.resolve_prefix_expr(self.db, prefix_expr)
    }

    fn resolve_index_expr(&self, index_expr: &ast::IndexExpr) -> Option<FunctionId> {
        self.analyze(index_expr.syntax())?.resolve_index_expr(self.db, index_expr)
    }

    fn resolve_bin_expr(&self, bin_expr: &ast::BinExpr) -> Option<FunctionId> {
        self.analyze(bin_expr.syntax())?.resolve_bin_expr(self.db, bin_expr)
    }

    fn resolve_try_expr(&self, try_expr: &ast::TryExpr) -> Option<FunctionId> {
        self.analyze(try_expr.syntax())?.resolve_try_expr(self.db, try_expr)
    }

    fn resolve_method_call_as_callable(&self, call: &ast::MethodCallExpr) -> Option<Callable> {
        self.analyze(call.syntax())?.resolve_method_call_as_callable(self.db, call)
    }

    fn resolve_field(&self, field: &ast::FieldExpr) -> Option<Field> {
        self.analyze(field.syntax())?.resolve_field(self.db, field)
    }

    fn resolve_record_field(
        &self,
        field: &ast::RecordExprField,
    ) -> Option<(Field, Option<Local>, Type)> {
        self.analyze(field.syntax())?.resolve_record_field(self.db, field)
    }

    fn resolve_record_pat_field(&self, field: &ast::RecordPatField) -> Option<(Field, Type)> {
        self.analyze(field.syntax())?.resolve_record_pat_field(self.db, field)
    }

    fn resolve_macro_call(&self, macro_call: &ast::MacroCall) -> Option<Macro> {
        let sa = self.analyze(macro_call.syntax())?;
        let macro_call = self.find_file(macro_call.syntax()).with_value(macro_call);
        sa.resolve_macro_call(self.db, macro_call)
    }

    fn is_unsafe_macro_call(&self, macro_call: &ast::MacroCall) -> bool {
        let sa = match self.analyze(macro_call.syntax()) {
            Some(it) => it,
            None => return false,
        };
        let macro_call = self.find_file(macro_call.syntax()).with_value(macro_call);
        sa.is_unsafe_macro_call(self.db, macro_call)
    }

    fn resolve_attr_macro_call(&self, item: &ast::Item) -> Option<Macro> {
        let item_in_file = self.wrap_node_infile(item.clone());
        let id = self.with_ctx(|ctx| {
            let macro_call_id = ctx.item_to_macro_call(item_in_file)?;
            macro_call_to_macro_id(ctx, self.db.upcast(), macro_call_id)
        })?;
        Some(Macro { id })
    }

    fn resolve_path(&self, path: &ast::Path) -> Option<PathResolution> {
        self.analyze(path.syntax())?.resolve_path(self.db, path)
    }

    fn resolve_extern_crate(&self, extern_crate: &ast::ExternCrate) -> Option<Crate> {
        let krate = self.scope(extern_crate.syntax())?.krate();
        let name = extern_crate.name_ref()?.as_name();
        if name == known::SELF_PARAM {
            return Some(krate);
        }
        krate
            .dependencies(self.db)
            .into_iter()
            .find_map(|dep| (dep.name == name).then_some(dep.krate))
    }

    fn resolve_variant(&self, record_lit: ast::RecordExpr) -> Option<VariantId> {
        self.analyze(record_lit.syntax())?.resolve_variant(self.db, record_lit)
    }

    fn resolve_bind_pat_to_const(&self, pat: &ast::IdentPat) -> Option<ModuleDef> {
        self.analyze(pat.syntax())?.resolve_bind_pat_to_const(self.db, pat)
    }

    fn record_literal_missing_fields(&self, literal: &ast::RecordExpr) -> Vec<(Field, Type)> {
        self.analyze(literal.syntax())
            .and_then(|it| it.record_literal_missing_fields(self.db, literal))
            .unwrap_or_default()
    }

    fn record_pattern_missing_fields(&self, pattern: &ast::RecordPat) -> Vec<(Field, Type)> {
        self.analyze(pattern.syntax())
            .and_then(|it| it.record_pattern_missing_fields(self.db, pattern))
            .unwrap_or_default()
    }

    fn with_ctx<F: FnOnce(&mut SourceToDefCtx<'_, '_>) -> T, T>(&self, f: F) -> T {
        let mut cache = self.s2d_cache.borrow_mut();
        let mut ctx = SourceToDefCtx { db: self.db, cache: &mut cache };
        f(&mut ctx)
    }

    fn to_def<T: ToDef>(&self, src: &T) -> Option<T::Def> {
        let src = self.find_file(src.syntax()).with_value(src).cloned();
        T::to_def(self, src)
    }

    fn to_module_def(&self, file: FileId) -> impl Iterator<Item = Module> {
        self.with_ctx(|ctx| ctx.file_to_def(file)).into_iter().map(Module::from)
    }

    fn scope(&self, node: &SyntaxNode) -> Option<SemanticsScope<'db>> {
        self.analyze_no_infer(node).map(|SourceAnalyzer { file_id, resolver, .. }| SemanticsScope {
            db: self.db,
            file_id,
            resolver,
        })
    }

    fn scope_at_offset(&self, node: &SyntaxNode, offset: TextSize) -> Option<SemanticsScope<'db>> {
        self.analyze_with_offset_no_infer(node, offset).map(
            |SourceAnalyzer { file_id, resolver, .. }| SemanticsScope {
                db: self.db,
                file_id,
                resolver,
            },
        )
    }

    fn scope_for_def(&self, def: Trait) -> SemanticsScope<'db> {
        let file_id = self.db.lookup_intern_trait(def.id).id.file_id();
        let resolver = def.id.resolver(self.db.upcast());
        SemanticsScope { db: self.db, file_id, resolver }
    }

    fn source<Def: HasSource>(&self, def: Def) -> Option<InFile<Def::Ast>>
    where
        Def::Ast: AstNode,
    {
        let res = def.source(self.db)?;
        self.cache(find_root(res.value.syntax()), res.file_id);
        Some(res)
    }

    /// Returns none if the file of the node is not part of a crate.
    fn analyze(&self, node: &SyntaxNode) -> Option<SourceAnalyzer> {
        self.analyze_impl(node, None, true)
    }

    /// Returns none if the file of the node is not part of a crate.
    fn analyze_no_infer(&self, node: &SyntaxNode) -> Option<SourceAnalyzer> {
        self.analyze_impl(node, None, false)
    }

    fn analyze_with_offset_no_infer(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> Option<SourceAnalyzer> {
        self.analyze_impl(node, Some(offset), false)
    }

    fn analyze_impl(
        &self,
        node: &SyntaxNode,
        offset: Option<TextSize>,
        infer_body: bool,
    ) -> Option<SourceAnalyzer> {
        let _p = profile::span("Semantics::analyze_impl");
        let node = self.find_file(node);

        let container = self.with_ctx(|ctx| ctx.find_container(node))?;

        let resolver = match container {
            ChildContainer::DefWithBodyId(def) => {
                return Some(if infer_body {
                    SourceAnalyzer::new_for_body(self.db, def, node, offset)
                } else {
                    SourceAnalyzer::new_for_body_no_infer(self.db, def, node, offset)
                })
            }
            ChildContainer::TraitId(it) => it.resolver(self.db.upcast()),
            ChildContainer::TraitAliasId(it) => it.resolver(self.db.upcast()),
            ChildContainer::ImplId(it) => it.resolver(self.db.upcast()),
            ChildContainer::ModuleId(it) => it.resolver(self.db.upcast()),
            ChildContainer::EnumId(it) => it.resolver(self.db.upcast()),
            ChildContainer::VariantId(it) => it.resolver(self.db.upcast()),
            ChildContainer::TypeAliasId(it) => it.resolver(self.db.upcast()),
            ChildContainer::GenericDefId(it) => it.resolver(self.db.upcast()),
        };
        Some(SourceAnalyzer::new_for_resolver(resolver, node))
    }

    fn cache(&self, root_node: SyntaxNode, file_id: HirFileId) {
        assert!(root_node.parent().is_none());
        let mut cache = self.cache.borrow_mut();
        let prev = cache.insert(root_node, file_id);
        assert!(prev == None || prev == Some(file_id))
    }

    fn assert_contains_node(&self, node: &SyntaxNode) {
        self.find_file(node);
    }

    fn lookup(&self, root_node: &SyntaxNode) -> Option<HirFileId> {
        let cache = self.cache.borrow();
        cache.get(root_node).copied()
    }

    fn wrap_node_infile<N: AstNode>(&self, node: N) -> InFile<N> {
        let InFile { file_id, .. } = self.find_file(node.syntax());
        InFile::new(file_id, node)
    }

    /// Wraps the node in a [`InFile`] with the file id it belongs to.
    fn find_file<'node>(&self, node: &'node SyntaxNode) -> InFile<&'node SyntaxNode> {
        let root_node = find_root(node);
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
                    .map(|it| format!("{it:?}"))
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
                let ty = self.type_of_expr(&field_expr.expr()?)?.original;
                if !ty.is_packed(self.db) {
                    return None;
                }

                let func = self.resolve_method_call(method_call_expr).map(Function::from)?;
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
            .map(|ty| ty.original.is_packed(self.db))
            .unwrap_or(false)

        // FIXME This needs layout computation to be correct. It will highlight
        // more than it should with the current implementation.
    }

    fn is_unsafe_ident_pat(&self, ident_pat: &ast::IdentPat) -> bool {
        if ident_pat.ref_token().is_none() {
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
            .map(|ty| ty.original.is_packed(self.db))
            .unwrap_or(false)
    }

    fn is_inside_unsafe(&self, expr: &ast::Expr) -> bool {
        let Some(enclosing_item) = expr.syntax().ancestors().find_map(Either::<ast::Item, ast::Variant>::cast) else { return false };

        let def = match &enclosing_item {
            Either::Left(ast::Item::Fn(it)) if it.unsafe_token().is_some() => return true,
            Either::Left(ast::Item::Fn(it)) => {
                self.to_def(it).map(<_>::into).map(DefWithBodyId::FunctionId)
            }
            Either::Left(ast::Item::Const(it)) => {
                self.to_def(it).map(<_>::into).map(DefWithBodyId::ConstId)
            }
            Either::Left(ast::Item::Static(it)) => {
                self.to_def(it).map(<_>::into).map(DefWithBodyId::StaticId)
            }
            Either::Left(_) => None,
            Either::Right(it) => self.to_def(it).map(<_>::into).map(DefWithBodyId::VariantId),
        };
        let Some(def) = def else { return false };
        let enclosing_node = enclosing_item.as_ref().either(|i| i.syntax(), |v| v.syntax());

        let (body, source_map) = self.db.body_with_source_map(def);

        let file_id = self.find_file(expr.syntax()).file_id;

        let Some(mut parent) = expr.syntax().parent() else { return false };
        loop {
            if &parent == enclosing_node {
                break false;
            }

            if let Some(parent) = ast::Expr::cast(parent.clone()) {
                if let Some(expr_id) = source_map.node_expr(InFile { file_id, value: &parent }) {
                    if let Expr::Unsafe { .. } = body[expr_id] {
                        break true;
                    }
                }
            }

            let Some(parent_) = parent.parent() else { break false };
            parent = parent_;
        }
    }
}

fn macro_call_to_macro_id(
    ctx: &mut SourceToDefCtx<'_, '_>,
    db: &dyn ExpandDatabase,
    macro_call_id: MacroCallId,
) -> Option<MacroId> {
    let loc = db.lookup_intern_macro_call(macro_call_id);
    match loc.def.kind {
        hir_expand::MacroDefKind::Declarative(it)
        | hir_expand::MacroDefKind::BuiltIn(_, it)
        | hir_expand::MacroDefKind::BuiltInAttr(_, it)
        | hir_expand::MacroDefKind::BuiltInDerive(_, it)
        | hir_expand::MacroDefKind::BuiltInEager(_, it) => {
            ctx.macro_to_def(InFile::new(it.file_id, it.to_node(db)))
        }
        hir_expand::MacroDefKind::ProcMacro(_, _, it) => {
            ctx.proc_macro_to_def(InFile::new(it.file_id, it.to_node(db)))
        }
    }
}

pub trait ToDef: AstNode + Clone {
    type Def;

    fn to_def(sema: &SemanticsImpl<'_>, src: InFile<Self>) -> Option<Self::Def>;
}

macro_rules! to_def_impls {
    ($(($def:path, $ast:path, $meth:ident)),* ,) => {$(
        impl ToDef for $ast {
            type Def = $def;
            fn to_def(sema: &SemanticsImpl<'_>, src: InFile<Self>) -> Option<Self::Def> {
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
    (crate::TraitAlias, ast::TraitAlias, trait_alias_to_def),
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
    (crate::GenericParam, ast::GenericParam, generic_param_to_def),
    (crate::Macro, ast::Macro, macro_to_def),
    (crate::Local, ast::IdentPat, bind_pat_to_def),
    (crate::Local, ast::SelfParam, self_param_to_def),
    (crate::Label, ast::Label, label_to_def),
    (crate::Adt, ast::Adt, adt_to_def),
];

fn find_root(node: &SyntaxNode) -> SyntaxNode {
    node.ancestors().last().unwrap()
}

/// `SemanticsScope` encapsulates the notion of a scope (the set of visible
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
    pub fn module(&self) -> Module {
        Module { id: self.resolver.module() }
    }

    pub fn krate(&self) -> Crate {
        Crate { id: self.resolver.krate() }
    }

    pub(crate) fn resolver(&self) -> &Resolver {
        &self.resolver
    }

    /// Note: `VisibleTraits` should be treated as an opaque type, passed into `Type
    pub fn visible_traits(&self) -> VisibleTraits {
        let resolver = &self.resolver;
        VisibleTraits(resolver.traits_in_scope(self.db.upcast()))
    }

    pub fn process_all_names(&self, f: &mut dyn FnMut(Name, ScopeDef)) {
        let scope = self.resolver.names_in_scope(self.db.upcast());
        for (name, entries) in scope {
            for entry in entries {
                let def = match entry {
                    resolver::ScopeDef::ModuleDef(it) => ScopeDef::ModuleDef(it.into()),
                    resolver::ScopeDef::Unknown => ScopeDef::Unknown,
                    resolver::ScopeDef::ImplSelfType(it) => ScopeDef::ImplSelfType(it.into()),
                    resolver::ScopeDef::AdtSelfType(it) => ScopeDef::AdtSelfType(it.into()),
                    resolver::ScopeDef::GenericParam(id) => ScopeDef::GenericParam(id.into()),
                    resolver::ScopeDef::Local(binding_id) => match self.resolver.body_owner() {
                        Some(parent) => ScopeDef::Local(Local { parent, binding_id }),
                        None => continue,
                    },
                    resolver::ScopeDef::Label(label_id) => match self.resolver.body_owner() {
                        Some(parent) => ScopeDef::Label(Label { parent, label_id }),
                        None => continue,
                    },
                };
                f(name.clone(), def)
            }
        }
    }

    /// Resolve a path as-if it was written at the given scope. This is
    /// necessary a heuristic, as it doesn't take hygiene into account.
    pub fn speculative_resolve(&self, path: &ast::Path) -> Option<PathResolution> {
        let ctx = body::LowerCtx::new(self.db.upcast(), self.file_id);
        let path = Path::from_src(path.clone(), &ctx)?;
        resolve_hir_path(self.db, &self.resolver, &path)
    }

    /// Iterates over associated types that may be specified after the given path (using
    /// `Ty::Assoc` syntax).
    pub fn assoc_type_shorthand_candidates<R>(
        &self,
        resolution: &PathResolution,
        mut cb: impl FnMut(&Name, TypeAlias) -> Option<R>,
    ) -> Option<R> {
        let def = self.resolver.generic_def()?;
        hir_ty::associated_type_shorthand_candidates(
            self.db,
            def,
            resolution.in_type_ns()?,
            |name, id| cb(name, id.into()),
        )
    }
}

#[derive(Debug)]
pub struct VisibleTraits(pub FxHashSet<TraitId>);

impl ops::Deref for VisibleTraits {
    type Target = FxHashSet<TraitId>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
