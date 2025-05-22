//! See `Semantics`.

mod child_by_source;
mod source_to_def;

use std::{
    cell::RefCell,
    convert::Infallible,
    fmt, iter, mem,
    ops::{self, ControlFlow, Not},
};

use either::Either;
use hir_def::{
    DefWithBodyId, FunctionId, MacroId, StructId, TraitId, VariantId,
    expr_store::{Body, ExprOrPatSource, path::Path},
    hir::{BindingId, Expr, ExprId, ExprOrPatId, Pat},
    nameres::{ModuleOrigin, crate_def_map},
    resolver::{self, HasResolver, Resolver, TypeNs},
    type_ref::Mutability,
};
use hir_expand::{
    EditionedFileId, ExpandResult, FileRange, HirFileId, InMacroFile, MacroCallId,
    attrs::collect_attrs,
    builtin::{BuiltinFnLikeExpander, EagerExpander},
    db::ExpandDatabase,
    files::{FileRangeWrapper, InRealFile},
    inert_attr_macro::find_builtin_attr_idx,
    mod_path::{ModPath, PathKind},
    name::AsName,
};
use hir_ty::diagnostics::unsafe_operations_for_body;
use intern::{Interned, Symbol, sym};
use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{SmallVec, smallvec};
use span::{Edition, FileId, SyntaxContext};
use stdx::TupleExt;
use syntax::{
    AstNode, AstToken, Direction, SyntaxKind, SyntaxNode, SyntaxNodePtr, SyntaxToken, TextRange,
    TextSize,
    algo::skip_trivia_token,
    ast::{self, HasAttrs as _, HasGenericParams},
};

use crate::{
    Adjust, Adjustment, Adt, AutoBorrow, BindingMode, BuiltinAttr, Callable, Const, ConstParam,
    Crate, DefWithBody, DeriveHelper, Enum, Field, Function, GenericSubstitution, HasSource, Impl,
    InFile, InlineAsmOperand, ItemInNs, Label, LifetimeParam, Local, Macro, Module, ModuleDef,
    Name, OverloadedDeref, ScopeDef, Static, Struct, ToolModule, Trait, TraitAlias, TupleField,
    Type, TypeAlias, TypeParam, Union, Variant, VariantDef,
    db::HirDatabase,
    semantics::source_to_def::{ChildContainer, SourceToDefCache, SourceToDefCtx},
    source_analyzer::{SourceAnalyzer, name_hygiene, resolve_hir_path},
};

const CONTINUE_NO_BREAKS: ControlFlow<Infallible, ()> = ControlFlow::Continue(());

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PathResolutionPerNs {
    pub type_ns: Option<PathResolution>,
    pub value_ns: Option<PathResolution>,
    pub macro_ns: Option<PathResolution>,
}

impl PathResolutionPerNs {
    pub fn new(
        type_ns: Option<PathResolution>,
        value_ns: Option<PathResolution>,
        macro_ns: Option<PathResolution>,
    ) -> Self {
        PathResolutionPerNs { type_ns, value_ns, macro_ns }
    }
    pub fn any(&self) -> Option<PathResolution> {
        self.type_ns.or(self.value_ns).or(self.macro_ns)
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
    /// MacroCall to its expansion's MacroCallId cache
    macro_call_cache: RefCell<FxHashMap<InFile<ast::MacroCall>, MacroCallId>>,
}

impl<DB> fmt::Debug for Semantics<'_, DB> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Semantics {{ ... }}")
    }
}

impl<'db, DB> ops::Deref for Semantics<'db, DB> {
    type Target = SemanticsImpl<'db>;

    fn deref(&self) -> &Self::Target {
        &self.imp
    }
}

impl<DB: HirDatabase> Semantics<'_, DB> {
    pub fn new(db: &DB) -> Semantics<'_, DB> {
        let impl_ = SemanticsImpl::new(db);
        Semantics { db, imp: impl_ }
    }

    pub fn hir_file_for(&self, syntax_node: &SyntaxNode) -> HirFileId {
        self.imp.find_file(syntax_node).file_id
    }

    pub fn token_ancestors_with_macros(
        &self,
        token: SyntaxToken,
    ) -> impl Iterator<Item = SyntaxNode> + '_ {
        token.parent().into_iter().flat_map(move |it| self.ancestors_with_macros(it))
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
    // FIXME: Rethink this API
    pub fn find_node_at_offset_with_descend<N: AstNode>(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> Option<N> {
        self.imp.descend_node_at_offset(node, offset).flatten().find_map(N::cast)
    }

    /// Find an AstNode by offset inside SyntaxNode, if it is inside an attribute macro call,
    /// descend it and find again
    // FIXME: Rethink this API
    pub fn find_nodes_at_offset_with_descend<'slf, N: AstNode + 'slf>(
        &'slf self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> impl Iterator<Item = N> + 'slf {
        self.imp.descend_node_at_offset(node, offset).filter_map(|mut it| it.find_map(N::cast))
    }

    pub fn resolve_range_pat(&self, range_pat: &ast::RangePat) -> Option<Struct> {
        self.imp.resolve_range_pat(range_pat).map(Struct::from)
    }

    pub fn resolve_range_expr(&self, range_expr: &ast::RangeExpr) -> Option<Struct> {
        self.imp.resolve_range_expr(range_expr).map(Struct::from)
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

    pub fn resolve_variant(&self, record_lit: ast::RecordExpr) -> Option<VariantDef> {
        self.imp.resolve_variant(record_lit).map(VariantDef::from)
    }

    pub fn file_to_module_def(&self, file: impl Into<FileId>) -> Option<Module> {
        self.imp.file_to_module_defs(file.into()).next()
    }

    pub fn file_to_module_defs(&self, file: impl Into<FileId>) -> impl Iterator<Item = Module> {
        self.imp.file_to_module_defs(file.into())
    }

    pub fn to_adt_def(&self, a: &ast::Adt) -> Option<Adt> {
        self.imp.to_def(a)
    }

    pub fn to_const_def(&self, c: &ast::Const) -> Option<Const> {
        self.imp.to_def(c)
    }

    pub fn to_enum_def(&self, e: &ast::Enum) -> Option<Enum> {
        self.imp.to_def(e)
    }

    pub fn to_enum_variant_def(&self, v: &ast::Variant) -> Option<Variant> {
        self.imp.to_def(v)
    }

    pub fn to_fn_def(&self, f: &ast::Fn) -> Option<Function> {
        self.imp.to_def(f)
    }

    pub fn to_impl_def(&self, i: &ast::Impl) -> Option<Impl> {
        self.imp.to_def(i)
    }

    pub fn to_macro_def(&self, m: &ast::Macro) -> Option<Macro> {
        self.imp.to_def(m)
    }

    pub fn to_module_def(&self, m: &ast::Module) -> Option<Module> {
        self.imp.to_def(m)
    }

    pub fn to_static_def(&self, s: &ast::Static) -> Option<Static> {
        self.imp.to_def(s)
    }

    pub fn to_struct_def(&self, s: &ast::Struct) -> Option<Struct> {
        self.imp.to_def(s)
    }

    pub fn to_trait_alias_def(&self, t: &ast::TraitAlias) -> Option<TraitAlias> {
        self.imp.to_def(t)
    }

    pub fn to_trait_def(&self, t: &ast::Trait) -> Option<Trait> {
        self.imp.to_def(t)
    }

    pub fn to_type_alias_def(&self, t: &ast::TypeAlias) -> Option<TypeAlias> {
        self.imp.to_def(t)
    }

    pub fn to_union_def(&self, u: &ast::Union) -> Option<Union> {
        self.imp.to_def(u)
    }
}

impl<'db> SemanticsImpl<'db> {
    fn new(db: &'db dyn HirDatabase) -> Self {
        SemanticsImpl { db, s2d_cache: Default::default(), macro_call_cache: Default::default() }
    }

    pub fn parse(&self, file_id: EditionedFileId) -> ast::SourceFile {
        let hir_file_id = file_id.into();
        let tree = self.db.parse(file_id).tree();
        self.cache(tree.syntax().clone(), hir_file_id);
        tree
    }

    /// If not crate is found for the file, try to return the last crate in topological order.
    pub fn first_crate(&self, file: FileId) -> Option<Crate> {
        match self.file_to_module_defs(file).next() {
            Some(module) => Some(module.krate()),
            None => self.db.all_crates().last().copied().map(Into::into),
        }
    }

    pub fn attach_first_edition(&self, file: FileId) -> Option<EditionedFileId> {
        Some(EditionedFileId::new(
            self.db,
            file,
            self.file_to_module_defs(file).next()?.krate().edition(self.db),
        ))
    }

    pub fn parse_guess_edition(&self, file_id: FileId) -> ast::SourceFile {
        let file_id = self
            .attach_first_edition(file_id)
            .unwrap_or_else(|| EditionedFileId::new(self.db, file_id, Edition::CURRENT));

        let tree = self.db.parse(file_id).tree();
        self.cache(tree.syntax().clone(), file_id.into());
        tree
    }

    pub fn find_parent_file(&self, file_id: HirFileId) -> Option<InFile<SyntaxNode>> {
        match file_id {
            HirFileId::FileId(file_id) => {
                let module = self.file_to_module_defs(file_id.file_id(self.db)).next()?;
                let def_map = crate_def_map(self.db, module.krate().id);
                match def_map[module.id.local_id].origin {
                    ModuleOrigin::CrateRoot { .. } => None,
                    ModuleOrigin::File { declaration, declaration_tree_id, .. } => {
                        let file_id = declaration_tree_id.file_id();
                        let in_file = InFile::new(file_id, declaration);
                        let node = in_file.to_node(self.db);
                        let root = find_root(node.syntax());
                        self.cache(root, file_id);
                        Some(in_file.with_value(node.syntax().clone()))
                    }
                    _ => unreachable!("FileId can only belong to a file module"),
                }
            }
            HirFileId::MacroFile(macro_file) => {
                let node = self.db.lookup_intern_macro_call(macro_file).to_node(self.db);
                let root = find_root(&node.value);
                self.cache(root, node.file_id);
                Some(node)
            }
        }
    }

    /// Returns the `SyntaxNode` of the module. If this is a file module, returns
    /// the `SyntaxNode` of the *definition* file, not of the *declaration*.
    pub fn module_definition_node(&self, module: Module) -> InFile<SyntaxNode> {
        let def_map = module.id.def_map(self.db);
        let definition = def_map[module.id.local_id].origin.definition_source(self.db);
        let definition = definition.map(|it| it.node());
        let root_node = find_root(&definition.value);
        self.cache(root_node, definition.file_id);
        definition
    }

    pub fn parse_or_expand(&self, file_id: HirFileId) -> SyntaxNode {
        let node = self.db.parse_or_expand(file_id);
        self.cache(node.clone(), file_id);
        node
    }

    pub fn expand(&self, file_id: MacroCallId) -> ExpandResult<SyntaxNode> {
        let res = self.db.parse_macro_expansion(file_id).map(|it| it.0.syntax_node());
        self.cache(res.value.clone(), file_id.into());
        res
    }

    pub fn expand_macro_call(&self, macro_call: &ast::MacroCall) -> Option<SyntaxNode> {
        let sa = self.analyze_no_infer(macro_call.syntax())?;

        let macro_call = InFile::new(sa.file_id, macro_call);
        let file_id = sa.expand(self.db, macro_call)?;

        let node = self.parse_or_expand(file_id.into());
        Some(node)
    }

    pub fn check_cfg_attr(&self, attr: &ast::TokenTree) -> Option<bool> {
        let file_id = self.find_file(attr.syntax()).file_id;
        let krate = match file_id {
            HirFileId::FileId(file_id) => {
                self.file_to_module_defs(file_id.file_id(self.db)).next()?.krate().id
            }
            HirFileId::MacroFile(macro_file) => self.db.lookup_intern_macro_call(macro_file).krate,
        };
        hir_expand::check_cfg_attr_value(self.db, attr, krate)
    }

    /// Expands the macro if it isn't one of the built-in ones that expand to custom syntax or dummy
    /// expansions.
    pub fn expand_allowed_builtins(
        &self,
        macro_call: &ast::MacroCall,
    ) -> Option<ExpandResult<SyntaxNode>> {
        let sa = self.analyze_no_infer(macro_call.syntax())?;

        let macro_call = InFile::new(sa.file_id, macro_call);
        let file_id = sa.expand(self.db, macro_call)?;
        let macro_call = self.db.lookup_intern_macro_call(file_id);

        let skip = matches!(
            macro_call.def.kind,
            hir_expand::MacroDefKind::BuiltIn(
                _,
                BuiltinFnLikeExpander::Column
                    | BuiltinFnLikeExpander::File
                    | BuiltinFnLikeExpander::ModulePath
                    | BuiltinFnLikeExpander::Asm
                    | BuiltinFnLikeExpander::GlobalAsm
                    | BuiltinFnLikeExpander::NakedAsm
                    | BuiltinFnLikeExpander::LogSyntax
                    | BuiltinFnLikeExpander::TraceMacros
                    | BuiltinFnLikeExpander::FormatArgs
                    | BuiltinFnLikeExpander::FormatArgsNl
                    | BuiltinFnLikeExpander::ConstFormatArgs,
            ) | hir_expand::MacroDefKind::BuiltInEager(_, EagerExpander::CompileError)
        );
        if skip {
            // these macros expand to custom builtin syntax and/or dummy things, no point in
            // showing these to the user
            return None;
        }

        let node = self.expand(file_id);
        Some(node)
    }

    /// If `item` has an attribute macro attached to it, expands it.
    pub fn expand_attr_macro(&self, item: &ast::Item) -> Option<ExpandResult<SyntaxNode>> {
        let src = self.wrap_node_infile(item.clone());
        let macro_call_id = self.with_ctx(|ctx| ctx.item_to_macro_call(src.as_ref()))?;
        Some(self.expand(macro_call_id))
    }

    pub fn expand_derive_as_pseudo_attr_macro(&self, attr: &ast::Attr) -> Option<SyntaxNode> {
        let adt = attr.syntax().parent().and_then(ast::Adt::cast)?;
        let src = self.wrap_node_infile(attr.clone());
        let call_id = self.with_ctx(|ctx| {
            ctx.attr_to_derive_macro_call(src.with_value(&adt), src).map(|(_, it, _)| it)
        })?;
        Some(self.parse_or_expand(call_id.into()))
    }

    pub fn resolve_derive_macro(&self, attr: &ast::Attr) -> Option<Vec<Option<Macro>>> {
        let calls = self.derive_macro_calls(attr)?;
        self.with_ctx(|ctx| {
            Some(
                calls
                    .into_iter()
                    .map(|call| macro_call_to_macro_id(ctx, call?).map(|id| Macro { id }))
                    .collect(),
            )
        })
    }

    pub fn expand_derive_macro(&self, attr: &ast::Attr) -> Option<Vec<ExpandResult<SyntaxNode>>> {
        let res: Vec<_> = self
            .derive_macro_calls(attr)?
            .into_iter()
            .flat_map(|call| {
                let file_id = call?;
                let ExpandResult { value, err } = self.db.parse_macro_expansion(file_id);
                let root_node = value.0.syntax_node();
                self.cache(root_node.clone(), file_id.into());
                Some(ExpandResult { value: root_node, err })
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

    pub fn is_derive_annotated(&self, adt: InFile<&ast::Adt>) -> bool {
        self.with_ctx(|ctx| ctx.has_derives(adt))
    }

    pub fn derive_helpers_in_scope(&self, adt: &ast::Adt) -> Option<Vec<(Symbol, Symbol)>> {
        let sa = self.analyze_no_infer(adt.syntax())?;
        let id = self.db.ast_id_map(sa.file_id).ast_id(adt);
        let result = sa
            .resolver
            .def_map()
            .derive_helpers_in_scope(InFile::new(sa.file_id, id))?
            .iter()
            .map(|(name, macro_, _)| {
                let macro_name = Macro::from(*macro_).name(self.db).symbol().clone();
                (name.symbol().clone(), macro_name)
            })
            .collect();
        Some(result)
    }

    pub fn derive_helper(&self, attr: &ast::Attr) -> Option<Vec<(Macro, MacroCallId)>> {
        let adt = attr.syntax().ancestors().find_map(ast::Item::cast).and_then(|it| match it {
            ast::Item::Struct(it) => Some(ast::Adt::Struct(it)),
            ast::Item::Enum(it) => Some(ast::Adt::Enum(it)),
            ast::Item::Union(it) => Some(ast::Adt::Union(it)),
            _ => None,
        })?;
        let attr_name = attr.path().and_then(|it| it.as_single_name_ref())?.as_name();
        let sa = self.analyze_no_infer(adt.syntax())?;
        let id = self.db.ast_id_map(sa.file_id).ast_id(&adt);
        let res: Vec<_> = sa
            .resolver
            .def_map()
            .derive_helpers_in_scope(InFile::new(sa.file_id, id))?
            .iter()
            .filter(|&(name, _, _)| *name == attr_name)
            .map(|&(_, macro_, call)| (macro_.into(), call))
            .collect();
        res.is_empty().not().then_some(res)
    }

    pub fn is_attr_macro_call(&self, item: InFile<&ast::Item>) -> bool {
        self.with_ctx(|ctx| ctx.item_to_macro_call(item).is_some())
    }

    /// Expand the macro call with a different token tree, mapping the `token_to_map` down into the
    /// expansion. `token_to_map` should be a token from the `speculative args` node.
    pub fn speculative_expand_macro_call(
        &self,
        actual_macro_call: &ast::MacroCall,
        speculative_args: &ast::TokenTree,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, Vec<(SyntaxToken, u8)>)> {
        let analyzer = self.analyze_no_infer(actual_macro_call.syntax())?;
        let macro_call = InFile::new(analyzer.file_id, actual_macro_call);
        let macro_file = analyzer.expansion(macro_call)?;
        hir_expand::db::expand_speculative(
            self.db,
            macro_file,
            speculative_args.syntax(),
            token_to_map,
        )
    }

    pub fn speculative_expand_raw(
        &self,
        macro_file: MacroCallId,
        speculative_args: &SyntaxNode,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, Vec<(SyntaxToken, u8)>)> {
        hir_expand::db::expand_speculative(self.db, macro_file, speculative_args, token_to_map)
    }

    /// Expand the macro call with a different item as the input, mapping the `token_to_map` down into the
    /// expansion. `token_to_map` should be a token from the `speculative args` node.
    pub fn speculative_expand_attr_macro(
        &self,
        actual_macro_call: &ast::Item,
        speculative_args: &ast::Item,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, Vec<(SyntaxToken, u8)>)> {
        let macro_call = self.wrap_node_infile(actual_macro_call.clone());
        let macro_call_id = self.with_ctx(|ctx| ctx.item_to_macro_call(macro_call.as_ref()))?;
        hir_expand::db::expand_speculative(
            self.db,
            macro_call_id,
            speculative_args.syntax(),
            token_to_map,
        )
    }

    pub fn speculative_expand_derive_as_pseudo_attr_macro(
        &self,
        actual_macro_call: &ast::Attr,
        speculative_args: &ast::Attr,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, Vec<(SyntaxToken, u8)>)> {
        let attr = self.wrap_node_infile(actual_macro_call.clone());
        let adt = actual_macro_call.syntax().parent().and_then(ast::Adt::cast)?;
        let macro_call_id = self.with_ctx(|ctx| {
            ctx.attr_to_derive_macro_call(attr.with_value(&adt), attr).map(|(_, it, _)| it)
        })?;
        hir_expand::db::expand_speculative(
            self.db,
            macro_call_id,
            speculative_args.syntax(),
            token_to_map,
        )
    }

    /// Checks if renaming `renamed` to `new_name` may introduce conflicts with other locals,
    /// and returns the conflicting locals.
    pub fn rename_conflicts(&self, to_be_renamed: &Local, new_name: &str) -> Vec<Local> {
        let body = self.db.body(to_be_renamed.parent);
        let resolver = to_be_renamed.parent.resolver(self.db);
        let starting_expr =
            body.binding_owners.get(&to_be_renamed.binding_id).copied().unwrap_or(body.body_expr);
        let mut visitor = RenameConflictsVisitor {
            body: &body,
            conflicts: FxHashSet::default(),
            db: self.db,
            new_name: Symbol::intern(new_name),
            old_name: to_be_renamed.name(self.db).symbol().clone(),
            owner: to_be_renamed.parent,
            to_be_renamed: to_be_renamed.binding_id,
            resolver,
        };
        visitor.rename_conflicts(starting_expr);
        visitor
            .conflicts
            .into_iter()
            .map(|binding_id| Local { parent: to_be_renamed.parent, binding_id })
            .collect()
    }

    /// Retrieves all the formatting parts of the format_args! (or `asm!`) template string.
    pub fn as_format_args_parts(
        &self,
        string: &ast::String,
    ) -> Option<Vec<(TextRange, Option<Either<PathResolution, InlineAsmOperand>>)>> {
        let string_start = string.syntax().text_range().start();
        let token = self.wrap_token_infile(string.syntax().clone()).into_real_file().ok()?;
        self.descend_into_macros_breakable(token, |token, _| {
            (|| {
                let token = token.value;
                let string = ast::String::cast(token)?;
                let literal =
                    string.syntax().parent().filter(|it| it.kind() == SyntaxKind::LITERAL)?;
                let parent = literal.parent()?;
                if let Some(format_args) = ast::FormatArgsExpr::cast(parent.clone()) {
                    let source_analyzer = self.analyze_no_infer(format_args.syntax())?;
                    let format_args = self.wrap_node_infile(format_args);
                    let res = source_analyzer
                        .as_format_args_parts(self.db, format_args.as_ref())?
                        .map(|(range, res)| (range + string_start, res.map(Either::Left)))
                        .collect();
                    Some(res)
                } else {
                    let asm = ast::AsmExpr::cast(parent)?;
                    let source_analyzer = self.analyze_no_infer(asm.syntax())?;
                    let line = asm.template().position(|it| *it.syntax() == literal)?;
                    let asm = self.wrap_node_infile(asm);
                    let (owner, (expr, asm_parts)) = source_analyzer.as_asm_parts(asm.as_ref())?;
                    let res = asm_parts
                        .get(line)?
                        .iter()
                        .map(|&(range, index)| {
                            (
                                range + string_start,
                                Some(Either::Right(InlineAsmOperand { owner, expr, index })),
                            )
                        })
                        .collect();
                    Some(res)
                }
            })()
            .map_or(ControlFlow::Continue(()), ControlFlow::Break)
        })
    }

    /// Retrieves the formatting part of the format_args! template string at the given offset.
    pub fn check_for_format_args_template(
        &self,
        original_token: SyntaxToken,
        offset: TextSize,
    ) -> Option<(TextRange, Option<Either<PathResolution, InlineAsmOperand>>)> {
        let string_start = original_token.text_range().start();
        let original_token = self.wrap_token_infile(original_token).into_real_file().ok()?;
        self.descend_into_macros_breakable(original_token, |token, _| {
            (|| {
                let token = token.value;
                self.resolve_offset_in_format_args(
                    ast::String::cast(token)?,
                    offset.checked_sub(string_start)?,
                )
                .map(|(range, res)| (range + string_start, res))
            })()
            .map_or(ControlFlow::Continue(()), ControlFlow::Break)
        })
    }

    fn resolve_offset_in_format_args(
        &self,
        string: ast::String,
        offset: TextSize,
    ) -> Option<(TextRange, Option<Either<PathResolution, InlineAsmOperand>>)> {
        debug_assert!(offset <= string.syntax().text_range().len());
        let literal = string.syntax().parent().filter(|it| it.kind() == SyntaxKind::LITERAL)?;
        let parent = literal.parent()?;
        if let Some(format_args) = ast::FormatArgsExpr::cast(parent.clone()) {
            let source_analyzer = &self.analyze_no_infer(format_args.syntax())?;
            let format_args = self.wrap_node_infile(format_args);
            source_analyzer
                .resolve_offset_in_format_args(self.db, format_args.as_ref(), offset)
                .map(|(range, res)| (range, res.map(Either::Left)))
        } else {
            let asm = ast::AsmExpr::cast(parent)?;
            let source_analyzer = &self.analyze_no_infer(asm.syntax())?;
            let line = asm.template().position(|it| *it.syntax() == literal)?;
            let asm = self.wrap_node_infile(asm);
            source_analyzer.resolve_offset_in_asm_template(asm.as_ref(), line, offset).map(
                |(owner, (expr, range, index))| {
                    (range, Some(Either::Right(InlineAsmOperand { owner, expr, index })))
                },
            )
        }
    }

    pub fn debug_hir_at(&self, token: SyntaxToken) -> Option<String> {
        self.analyze_no_infer(&token.parent()?).and_then(|it| {
            Some(match it.body_or_sig.as_ref()? {
                crate::source_analyzer::BodyOrSig::Body { def, body, .. } => {
                    hir_def::expr_store::pretty::print_body_hir(
                        self.db,
                        body,
                        *def,
                        it.file_id.edition(self.db),
                    )
                }
                &crate::source_analyzer::BodyOrSig::VariantFields { def, .. } => {
                    hir_def::expr_store::pretty::print_variant_body_hir(
                        self.db,
                        def,
                        it.file_id.edition(self.db),
                    )
                }
                &crate::source_analyzer::BodyOrSig::Sig { def, .. } => {
                    hir_def::expr_store::pretty::print_signature(
                        self.db,
                        def,
                        it.file_id.edition(self.db),
                    )
                }
            })
        })
    }

    /// Maps a node down by mapping its first and last token down.
    pub fn descend_node_into_attributes<N: AstNode>(&self, node: N) -> SmallVec<[N; 1]> {
        // This might not be the correct way to do this, but it works for now
        let mut res = smallvec![];
        let tokens = (|| {
            // FIXME: the trivia skipping should not be necessary
            let first = skip_trivia_token(node.syntax().first_token()?, Direction::Next)?;
            let last = skip_trivia_token(node.syntax().last_token()?, Direction::Prev)?;
            Some((first, last))
        })();
        let (first, last) = match tokens {
            Some(it) => it,
            None => return res,
        };
        let file = self.find_file(node.syntax());
        let Some(file_id) = file.file_id.file_id() else {
            return res;
        };

        if first == last {
            // node is just the token, so descend the token
            self.descend_into_macros_impl(
                InRealFile::new(file_id, first),
                &mut |InFile { value, .. }, _ctx| {
                    if let Some(node) = value
                        .parent_ancestors()
                        .take_while(|it| it.text_range() == value.text_range())
                        .find_map(N::cast)
                    {
                        res.push(node)
                    }
                    CONTINUE_NO_BREAKS
                },
            );
        } else {
            // Descend first and last token, then zip them to look for the node they belong to
            let mut scratch: SmallVec<[_; 1]> = smallvec![];
            self.descend_into_macros_impl(InRealFile::new(file_id, first), &mut |token, _ctx| {
                scratch.push(token);
                CONTINUE_NO_BREAKS
            });

            let mut scratch = scratch.into_iter();
            self.descend_into_macros_impl(
                InRealFile::new(file_id, last),
                &mut |InFile { value: last, file_id: last_fid }, _ctx| {
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
                    CONTINUE_NO_BREAKS
                },
            );
        }
        res
    }

    // FIXME: This isn't quite right wrt to inner attributes
    /// Does a syntactic traversal to check whether this token might be inside a macro call
    pub fn might_be_inside_macro_call(&self, token: &SyntaxToken) -> bool {
        token.parent_ancestors().any(|ancestor| {
            if ast::MacroCall::can_cast(ancestor.kind()) {
                return true;
            }
            // Check if it is an item (only items can have macro attributes) that has a non-builtin attribute.
            let Some(item) = ast::Item::cast(ancestor) else { return false };
            item.attrs().any(|attr| {
                let Some(meta) = attr.meta() else { return false };
                let Some(path) = meta.path() else { return false };
                if let Some(attr_name) = path.as_single_name_ref() {
                    let attr_name = attr_name.text();
                    let attr_name = Symbol::intern(attr_name.as_str());
                    if attr_name == sym::derive {
                        return true;
                    }
                    // We ignore `#[test]` and friends in the def map, so we cannot expand them.
                    // FIXME: We match by text. This is both hacky and incorrect (people can, and do, create
                    // other macros named `test`). We cannot fix that unfortunately because we use this method
                    // for speculative expansion in completion, which we cannot analyze. Fortunately, most macros
                    // named `test` are test-like, meaning their expansion is not terribly important for IDE.
                    if attr_name == sym::test
                        || attr_name == sym::bench
                        || attr_name == sym::test_case
                        || find_builtin_attr_idx(&attr_name).is_some()
                    {
                        return false;
                    }
                }
                let mut segments = path.segments();
                let mut next_segment_text = || segments.next().and_then(|it| it.name_ref());
                // `#[core::prelude::rust_2024::test]` or `#[std::prelude::rust_2024::test]`.
                if next_segment_text().is_some_and(|it| matches!(&*it.text(), "core" | "std"))
                    && next_segment_text().is_some_and(|it| it.text() == "prelude")
                    && next_segment_text().is_some()
                    && next_segment_text()
                        .is_some_and(|it| matches!(&*it.text(), "test" | "bench" | "test_case"))
                {
                    return false;
                }
                true
            })
        })
    }

    pub fn descend_into_macros_cb(
        &self,
        token: SyntaxToken,
        mut cb: impl FnMut(InFile<SyntaxToken>, SyntaxContext),
    ) {
        if let Ok(token) = self.wrap_token_infile(token).into_real_file() {
            self.descend_into_macros_impl(token, &mut |t, ctx| {
                cb(t, ctx);
                CONTINUE_NO_BREAKS
            });
        }
    }

    pub fn descend_into_macros(&self, token: SyntaxToken) -> SmallVec<[SyntaxToken; 1]> {
        let mut res = smallvec![];
        if let Ok(token) = self.wrap_token_infile(token.clone()).into_real_file() {
            self.descend_into_macros_impl(token, &mut |t, _ctx| {
                res.push(t.value);
                CONTINUE_NO_BREAKS
            });
        }
        if res.is_empty() {
            res.push(token);
        }
        res
    }

    pub fn descend_into_macros_no_opaque(
        &self,
        token: SyntaxToken,
    ) -> SmallVec<[InFile<SyntaxToken>; 1]> {
        let mut res = smallvec![];
        let token = self.wrap_token_infile(token);
        if let Ok(token) = token.clone().into_real_file() {
            self.descend_into_macros_impl(token, &mut |t, ctx| {
                if !ctx.is_opaque(self.db) {
                    // Don't descend into opaque contexts
                    res.push(t);
                }
                CONTINUE_NO_BREAKS
            });
        }
        if res.is_empty() {
            res.push(token);
        }
        res
    }

    pub fn descend_into_macros_breakable<T>(
        &self,
        token: InRealFile<SyntaxToken>,
        mut cb: impl FnMut(InFile<SyntaxToken>, SyntaxContext) -> ControlFlow<T>,
    ) -> Option<T> {
        self.descend_into_macros_impl(token, &mut cb)
    }

    /// Descends the token into expansions, returning the tokens that matches the input
    /// token's [`SyntaxKind`] and text.
    pub fn descend_into_macros_exact(&self, token: SyntaxToken) -> SmallVec<[SyntaxToken; 1]> {
        let mut r = smallvec![];
        let text = token.text();
        let kind = token.kind();

        self.descend_into_macros_cb(token.clone(), |InFile { value, file_id: _ }, ctx| {
            let mapped_kind = value.kind();
            let any_ident_match = || kind.is_any_identifier() && value.kind().is_any_identifier();
            let matches = (kind == mapped_kind || any_ident_match())
                && text == value.text()
                && !ctx.is_opaque(self.db);
            if matches {
                r.push(value);
            }
        });
        if r.is_empty() {
            r.push(token);
        }
        r
    }

    /// Descends the token into expansions, returning the first token that matches the input
    /// token's [`SyntaxKind`] and text.
    pub fn descend_into_macros_single_exact(&self, token: SyntaxToken) -> SyntaxToken {
        let text = token.text();
        let kind = token.kind();
        if let Ok(token) = self.wrap_token_infile(token.clone()).into_real_file() {
            self.descend_into_macros_breakable(token, |InFile { value, file_id: _ }, _ctx| {
                let mapped_kind = value.kind();
                let any_ident_match =
                    || kind.is_any_identifier() && value.kind().is_any_identifier();
                let matches = (kind == mapped_kind || any_ident_match()) && text == value.text();
                if matches { ControlFlow::Break(value) } else { ControlFlow::Continue(()) }
            })
        } else {
            None
        }
        .unwrap_or(token)
    }

    fn descend_into_macros_impl<T>(
        &self,
        InRealFile { value: token, file_id }: InRealFile<SyntaxToken>,
        f: &mut dyn FnMut(InFile<SyntaxToken>, SyntaxContext) -> ControlFlow<T>,
    ) -> Option<T> {
        let _p = tracing::info_span!("descend_into_macros_impl").entered();

        let span = self.db.real_span_map(file_id).span_for_range(token.text_range());

        // Process the expansion of a call, pushing all tokens with our span in the expansion back onto our stack
        let process_expansion_for_token = |stack: &mut Vec<_>, macro_file| {
            let InMacroFile { file_id, value: mapped_tokens } = self.with_ctx(|ctx| {
                Some(
                    ctx.cache
                        .get_or_insert_expansion(ctx.db, macro_file)
                        .map_range_down(span)?
                        .map(SmallVec::<[_; 2]>::from_iter),
                )
            })?;
            // we have found a mapping for the token if the vec is non-empty
            let res = mapped_tokens.is_empty().not().then_some(());
            // requeue the tokens we got from mapping our current token down
            stack.push((HirFileId::from(file_id), mapped_tokens));
            res
        };

        // A stack of tokens to process, along with the file they came from
        // These are tracked to know which macro calls we still have to look into
        // the tokens themselves aren't that interesting as the span that is being used to map
        // things down never changes.
        let mut stack: Vec<(_, SmallVec<[_; 2]>)> = vec![];
        let include = self.s2d_cache.borrow_mut().get_or_insert_include_for(self.db, file_id);
        match include {
            Some(include) => {
                // include! inputs are always from real files, so they only need to be handled once upfront
                process_expansion_for_token(&mut stack, include)?;
            }
            None => {
                stack.push((
                    file_id.into(),
                    smallvec![(token, SyntaxContext::root(file_id.edition(self.db)))],
                ));
            }
        }

        let mut m_cache = self.macro_call_cache.borrow_mut();

        // Filters out all tokens that contain the given range (usually the macro call), any such
        // token is redundant as the corresponding macro call has already been processed
        let filter_duplicates = |tokens: &mut SmallVec<_>, range: TextRange| {
            tokens.retain(|(t, _): &mut (SyntaxToken, _)| !range.contains_range(t.text_range()))
        };

        while let Some((expansion, ref mut tokens)) = stack.pop() {
            // Reverse the tokens so we prefer first tokens (to accommodate for popping from the
            // back)
            // alternatively we could pop from the front but that would shift the content on every pop
            tokens.reverse();
            while let Some((token, ctx)) = tokens.pop() {
                let was_not_remapped = (|| {
                    // First expand into attribute invocations
                    let containing_attribute_macro_call = self.with_ctx(|ctx| {
                        token.parent_ancestors().filter_map(ast::Item::cast).find_map(|item| {
                            // Don't force populate the dyn cache for items that don't have an attribute anyways
                            item.attrs().next()?;
                            Some((ctx.item_to_macro_call(InFile::new(expansion, &item))?, item))
                        })
                    });
                    if let Some((call_id, item)) = containing_attribute_macro_call {
                        let attr_id = match self.db.lookup_intern_macro_call(call_id).kind {
                            hir_expand::MacroCallKind::Attr { invoc_attr_index, .. } => {
                                invoc_attr_index.ast_index()
                            }
                            _ => 0,
                        };
                        // FIXME: here, the attribute's text range is used to strip away all
                        // entries from the start of the attribute "list" up the invoking
                        // attribute. But in
                        // ```
                        // mod foo {
                        //     #![inner]
                        // }
                        // ```
                        // we don't wanna strip away stuff in the `mod foo {` range, that is
                        // here if the id corresponds to an inner attribute we got strip all
                        // text ranges of the outer ones, and then all of the inner ones up
                        // to the invoking attribute so that the inbetween is ignored.
                        let text_range = item.syntax().text_range();
                        let start = collect_attrs(&item)
                            .nth(attr_id)
                            .map(|attr| match attr.1 {
                                Either::Left(it) => it.syntax().text_range().start(),
                                Either::Right(it) => it.syntax().text_range().start(),
                            })
                            .unwrap_or_else(|| text_range.start());
                        let text_range = TextRange::new(start, text_range.end());
                        filter_duplicates(tokens, text_range);
                        return process_expansion_for_token(&mut stack, call_id);
                    }

                    // Then check for token trees, that means we are either in a function-like macro or
                    // secondary attribute inputs
                    let tt = token
                        .parent_ancestors()
                        .map_while(Either::<ast::TokenTree, ast::Meta>::cast)
                        .last()?;
                    match tt {
                        // function-like macro call
                        Either::Left(tt) => {
                            if tt.left_delimiter_token().map_or(false, |it| it == token) {
                                return None;
                            }
                            if tt.right_delimiter_token().map_or(false, |it| it == token) {
                                return None;
                            }
                            let macro_call = tt.syntax().parent().and_then(ast::MacroCall::cast)?;
                            let mcall = InFile::new(expansion, macro_call);
                            let file_id = match m_cache.get(&mcall) {
                                Some(&it) => it,
                                None => {
                                    let it = token
                                        .parent()
                                        .and_then(|parent| {
                                            self.analyze_impl(
                                                InFile::new(expansion, &parent),
                                                None,
                                                false,
                                            )
                                        })?
                                        .expand(self.db, mcall.as_ref())?;
                                    m_cache.insert(mcall, it);
                                    it
                                }
                            };
                            let text_range = tt.syntax().text_range();
                            filter_duplicates(tokens, text_range);

                            process_expansion_for_token(&mut stack, file_id).or(file_id
                                .eager_arg(self.db)
                                .and_then(|arg| {
                                    // also descend into eager expansions
                                    process_expansion_for_token(&mut stack, arg)
                                }))
                        }
                        // derive or derive helper
                        Either::Right(meta) => {
                            // attribute we failed expansion for earlier, this might be a derive invocation
                            // or derive helper attribute
                            let attr = meta.parent_attr()?;
                            let adt = match attr.syntax().parent().and_then(ast::Adt::cast) {
                                Some(adt) => {
                                    // this might be a derive on an ADT
                                    let derive_call = self.with_ctx(|ctx| {
                                        // so try downmapping the token into the pseudo derive expansion
                                        // see [hir_expand::builtin_attr_macro] for how the pseudo derive expansion works
                                        ctx.attr_to_derive_macro_call(
                                            InFile::new(expansion, &adt),
                                            InFile::new(expansion, attr.clone()),
                                        )
                                        .map(|(_, call_id, _)| call_id)
                                    });

                                    match derive_call {
                                        Some(call_id) => {
                                            // resolved to a derive
                                            let text_range = attr.syntax().text_range();
                                            // remove any other token in this macro input, all their mappings are the
                                            // same as this
                                            tokens.retain(|(t, _)| {
                                                !text_range.contains_range(t.text_range())
                                            });
                                            return process_expansion_for_token(
                                                &mut stack, call_id,
                                            );
                                        }
                                        None => Some(adt),
                                    }
                                }
                                None => {
                                    // Otherwise this could be a derive helper on a variant or field
                                    attr.syntax().ancestors().find_map(ast::Item::cast).and_then(
                                        |it| match it {
                                            ast::Item::Struct(it) => Some(ast::Adt::Struct(it)),
                                            ast::Item::Enum(it) => Some(ast::Adt::Enum(it)),
                                            ast::Item::Union(it) => Some(ast::Adt::Union(it)),
                                            _ => None,
                                        },
                                    )
                                }
                            }?;
                            if !self.with_ctx(|ctx| ctx.has_derives(InFile::new(expansion, &adt))) {
                                return None;
                            }
                            let attr_name =
                                attr.path().and_then(|it| it.as_single_name_ref())?.as_name();
                            // Not an attribute, nor a derive, so it's either an intert attribute or a derive helper
                            // Try to resolve to a derive helper and downmap
                            let resolver = &token
                                .parent()
                                .and_then(|parent| {
                                    self.analyze_impl(InFile::new(expansion, &parent), None, false)
                                })?
                                .resolver;
                            let id = self.db.ast_id_map(expansion).ast_id(&adt);
                            let helpers = resolver
                                .def_map()
                                .derive_helpers_in_scope(InFile::new(expansion, id))?;

                            if !helpers.is_empty() {
                                let text_range = attr.syntax().text_range();
                                filter_duplicates(tokens, text_range);
                            }

                            let mut res = None;
                            for (.., derive) in
                                helpers.iter().filter(|(helper, ..)| *helper == attr_name)
                            {
                                // as there may be multiple derives registering the same helper
                                // name, we gotta make sure to call this for all of them!
                                // FIXME: We need to call `f` for all of them as well though!
                                res = res.or(process_expansion_for_token(&mut stack, *derive));
                            }
                            res
                        }
                    }
                })()
                .is_none();

                if was_not_remapped {
                    if let ControlFlow::Break(b) = f(InFile::new(expansion, token), ctx) {
                        return Some(b);
                    }
                }
            }
        }
        None
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
            .map(move |token| self.descend_into_macros_exact(token))
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

    /// Attempts to map the node out of macro expanded files returning the original file range.
    /// If upmapping is not possible, this will fall back to the range of the macro call of the
    /// macro file the node resides in.
    pub fn original_range(&self, node: &SyntaxNode) -> FileRange {
        let node = self.find_file(node);
        node.original_file_range_rooted(self.db)
    }

    /// Attempts to map the node out of macro expanded files returning the original file range.
    pub fn original_range_opt(&self, node: &SyntaxNode) -> Option<FileRange> {
        let node = self.find_file(node);
        node.original_file_range_opt(self.db).filter(|(_, ctx)| ctx.is_root()).map(TupleExt::head)
    }

    /// Attempts to map the node out of macro expanded files.
    /// This only work for attribute expansions, as other ones do not have nodes as input.
    pub fn original_ast_node<N: AstNode>(&self, node: N) -> Option<N> {
        self.wrap_node_infile(node).original_ast_node_rooted(self.db).map(
            |InRealFile { file_id, value }| {
                self.cache(find_root(value.syntax()), file_id.into());
                value
            },
        )
    }

    /// Attempts to map the node out of macro expanded files.
    /// This only work for attribute expansions, as other ones do not have nodes as input.
    pub fn original_syntax_node_rooted(&self, node: &SyntaxNode) -> Option<SyntaxNode> {
        let InFile { file_id, .. } = self.find_file(node);
        InFile::new(file_id, node).original_syntax_node_rooted(self.db).map(
            |InRealFile { file_id, value }| {
                self.cache(find_root(&value), file_id.into());
                value
            },
        )
    }

    pub fn diagnostics_display_range(
        &self,
        src: InFile<SyntaxNodePtr>,
    ) -> FileRangeWrapper<FileId> {
        let root = self.parse_or_expand(src.file_id);
        let node = src.map(|it| it.to_node(&root));
        let FileRange { file_id, range } = node.as_ref().original_file_range_rooted(self.db);
        FileRangeWrapper { file_id: file_id.file_id(self.db), range }
    }

    fn token_ancestors_with_macros(
        &self,
        token: SyntaxToken,
    ) -> impl Iterator<Item = SyntaxNode> + Clone + '_ {
        token.parent().into_iter().flat_map(move |parent| self.ancestors_with_macros(parent))
    }

    /// Iterates the ancestors of the given node, climbing up macro expansions while doing so.
    pub fn ancestors_with_macros(
        &self,
        node: SyntaxNode,
    ) -> impl Iterator<Item = SyntaxNode> + Clone + '_ {
        let node = self.find_file(&node);
        iter::successors(Some(node.cloned()), move |&InFile { file_id, ref value }| {
            match value.parent() {
                Some(parent) => Some(InFile::new(file_id, parent)),
                None => {
                    let macro_file = file_id.macro_file()?;

                    self.with_ctx(|ctx| {
                        let expansion_info = ctx.cache.get_or_insert_expansion(ctx.db, macro_file);
                        expansion_info.arg().map(|node| node?.parent()).transpose()
                    })
                }
            }
        })
        .map(|it| it.value)
    }

    pub fn ancestors_at_offset_with_macros(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> impl Iterator<Item = SyntaxNode> + '_ {
        node.token_at_offset(offset)
            .map(|token| self.token_ancestors_with_macros(token))
            .kmerge_by(|node1, node2| node1.text_range().len() < node2.text_range().len())
    }

    pub fn resolve_lifetime_param(&self, lifetime: &ast::Lifetime) -> Option<LifetimeParam> {
        let text = lifetime.text();
        let lifetime_param = lifetime.syntax().ancestors().find_map(|syn| {
            let gpl = ast::AnyHasGenericParams::cast(syn)?.generic_param_list()?;
            gpl.lifetime_params()
                .find(|tp| tp.lifetime().as_ref().map(|lt| lt.text()).as_ref() == Some(&text))
        })?;
        let src = self.wrap_node_infile(lifetime_param);
        ToDef::to_def(self, src.as_ref())
    }

    pub fn resolve_label(&self, label: &ast::Lifetime) -> Option<Label> {
        let src = self.wrap_node_infile(label.clone());
        let (parent, label_id) = self.with_ctx(|ctx| ctx.label_ref_to_def(src.as_ref()))?;
        Some(Label { parent, label_id })
    }

    pub fn resolve_type(&self, ty: &ast::Type) -> Option<Type> {
        let analyze = self.analyze(ty.syntax())?;
        analyze.type_of_type(self.db, ty)
    }

    pub fn resolve_trait(&self, path: &ast::Path) -> Option<Trait> {
        let parent_ty = path.syntax().parent().and_then(ast::Type::cast)?;
        let analyze = self.analyze(path.syntax())?;
        let ty = analyze.store_sm()?.node_type(InFile::new(analyze.file_id, &parent_ty))?;
        let path = match &analyze.store()?.types[ty] {
            hir_def::type_ref::TypeRef::Path(path) => path,
            _ => return None,
        };
        match analyze.resolver.resolve_path_in_type_ns_fully(self.db, path)? {
            TypeNs::TraitId(trait_id) => Some(trait_id.into()),
            _ => None,
        }
    }

    pub fn expr_adjustments(&self, expr: &ast::Expr) -> Option<Vec<Adjustment>> {
        let mutability = |m| match m {
            hir_ty::Mutability::Not => Mutability::Shared,
            hir_ty::Mutability::Mut => Mutability::Mut,
        };

        let analyzer = self.analyze(expr.syntax())?;

        let (mut source_ty, _) = analyzer.type_of_expr(self.db, expr)?;

        analyzer.expr_adjustments(expr).map(|it| {
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
                        hir_ty::Adjust::Borrow(hir_ty::AutoBorrow::Ref(_, m)) => {
                            // FIXME: Handle lifetimes here
                            Adjust::Borrow(AutoBorrow::Ref(mutability(m)))
                        }
                        hir_ty::Adjust::Pointer(pc) => Adjust::Pointer(pc),
                    };

                    // Update `source_ty` for the next adjustment
                    let source = mem::replace(&mut source_ty, target.clone());

                    Adjustment { source, target, kind }
                })
                .collect()
        })
    }

    pub fn type_of_expr(&self, expr: &ast::Expr) -> Option<TypeInfo> {
        self.analyze(expr.syntax())?
            .type_of_expr(self.db, expr)
            .map(|(ty, coerced)| TypeInfo { original: ty, adjusted: coerced })
    }

    pub fn type_of_pat(&self, pat: &ast::Pat) -> Option<TypeInfo> {
        self.analyze(pat.syntax())?
            .type_of_pat(self.db, pat)
            .map(|(ty, coerced)| TypeInfo { original: ty, adjusted: coerced })
    }

    /// It also includes the changes that binding mode makes in the type. For example in
    /// `let ref x @ Some(_) = None` the result of `type_of_pat` is `Option<T>` but the result
    /// of this function is `&mut Option<T>`
    pub fn type_of_binding_in_pat(&self, pat: &ast::IdentPat) -> Option<Type> {
        self.analyze(pat.syntax())?.type_of_binding_in_pat(self.db, pat)
    }

    pub fn type_of_self(&self, param: &ast::SelfParam) -> Option<Type> {
        self.analyze(param.syntax())?.type_of_self(self.db, param)
    }

    pub fn pattern_adjustments(&self, pat: &ast::Pat) -> SmallVec<[Type; 1]> {
        self.analyze(pat.syntax())
            .and_then(|it| it.pattern_adjustments(self.db, pat))
            .unwrap_or_default()
    }

    pub fn binding_mode_of_pat(&self, pat: &ast::IdentPat) -> Option<BindingMode> {
        self.analyze(pat.syntax())?.binding_mode_of_pat(self.db, pat)
    }

    pub fn resolve_expr_as_callable(&self, call: &ast::Expr) -> Option<Callable> {
        self.analyze(call.syntax())?.resolve_expr_as_callable(self.db, call)
    }

    pub fn resolve_method_call(&self, call: &ast::MethodCallExpr) -> Option<Function> {
        self.analyze(call.syntax())?.resolve_method_call(self.db, call)
    }

    /// Attempts to resolve this call expression as a method call falling back to resolving it as a field.
    pub fn resolve_method_call_fallback(
        &self,
        call: &ast::MethodCallExpr,
    ) -> Option<(Either<Function, Field>, Option<GenericSubstitution>)> {
        self.analyze(call.syntax())?.resolve_method_call_fallback(self.db, call)
    }

    /// Env is used to derive the trait environment
    // FIXME: better api for the trait environment
    pub fn resolve_trait_impl_method(
        &self,
        env: Type,
        trait_: Trait,
        func: Function,
        subst: impl IntoIterator<Item = Type>,
    ) -> Option<Function> {
        let mut substs = hir_ty::TyBuilder::subst_for_def(self.db, TraitId::from(trait_), None);
        for s in subst {
            substs = substs.push(s.ty);
        }
        Some(self.db.lookup_impl_method(env.env, func.into(), substs.build()).0.into())
    }

    fn resolve_range_pat(&self, range_pat: &ast::RangePat) -> Option<StructId> {
        self.analyze(range_pat.syntax())?.resolve_range_pat(self.db, range_pat)
    }

    fn resolve_range_expr(&self, range_expr: &ast::RangeExpr) -> Option<StructId> {
        self.analyze(range_expr.syntax())?.resolve_range_expr(self.db, range_expr)
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

    // This does not resolve the method call to the correct trait impl!
    // We should probably fix that.
    pub fn resolve_method_call_as_callable(&self, call: &ast::MethodCallExpr) -> Option<Callable> {
        self.analyze(call.syntax())?.resolve_method_call_as_callable(self.db, call)
    }

    pub fn resolve_field(&self, field: &ast::FieldExpr) -> Option<Either<Field, TupleField>> {
        self.analyze(field.syntax())?.resolve_field(field)
    }

    pub fn resolve_field_fallback(
        &self,
        field: &ast::FieldExpr,
    ) -> Option<(Either<Either<Field, TupleField>, Function>, Option<GenericSubstitution>)> {
        self.analyze(field.syntax())?.resolve_field_fallback(self.db, field)
    }

    pub fn resolve_record_field(
        &self,
        field: &ast::RecordExprField,
    ) -> Option<(Field, Option<Local>, Type)> {
        self.resolve_record_field_with_substitution(field)
            .map(|(field, local, ty, _)| (field, local, ty))
    }

    pub fn resolve_record_field_with_substitution(
        &self,
        field: &ast::RecordExprField,
    ) -> Option<(Field, Option<Local>, Type, GenericSubstitution)> {
        self.analyze(field.syntax())?.resolve_record_field(self.db, field)
    }

    pub fn resolve_record_pat_field(&self, field: &ast::RecordPatField) -> Option<(Field, Type)> {
        self.resolve_record_pat_field_with_subst(field).map(|(field, ty, _)| (field, ty))
    }

    pub fn resolve_record_pat_field_with_subst(
        &self,
        field: &ast::RecordPatField,
    ) -> Option<(Field, Type, GenericSubstitution)> {
        self.analyze(field.syntax())?.resolve_record_pat_field(self.db, field)
    }

    // FIXME: Replace this with `resolve_macro_call2`
    pub fn resolve_macro_call(&self, macro_call: &ast::MacroCall) -> Option<Macro> {
        let macro_call = self.find_file(macro_call.syntax()).with_value(macro_call);
        self.resolve_macro_call2(macro_call)
    }

    pub fn resolve_macro_call2(&self, macro_call: InFile<&ast::MacroCall>) -> Option<Macro> {
        self.with_ctx(|ctx| {
            ctx.macro_call_to_macro_call(macro_call)
                .and_then(|call| macro_call_to_macro_id(ctx, call))
                .map(Into::into)
        })
        .or_else(|| {
            self.analyze(macro_call.value.syntax())?.resolve_macro_call(self.db, macro_call)
        })
    }

    pub fn is_proc_macro_call(&self, macro_call: InFile<&ast::MacroCall>) -> bool {
        self.resolve_macro_call2(macro_call)
            .is_some_and(|m| matches!(m.id, MacroId::ProcMacroId(..)))
    }

    pub fn resolve_macro_call_arm(&self, macro_call: &ast::MacroCall) -> Option<u32> {
        let sa = self.analyze(macro_call.syntax())?;
        self.db
            .parse_macro_expansion(
                sa.expand(self.db, self.wrap_node_infile(macro_call.clone()).as_ref())?,
            )
            .value
            .1
            .matched_arm
    }

    pub fn get_unsafe_ops(&self, def: DefWithBody) -> FxHashSet<ExprOrPatSource> {
        let def = DefWithBodyId::from(def);
        let (body, source_map) = self.db.body_with_source_map(def);
        let infer = self.db.infer(def);
        let mut res = FxHashSet::default();
        unsafe_operations_for_body(self.db, &infer, def, &body, &mut |node| {
            if let Ok(node) = source_map.expr_or_pat_syntax(node) {
                res.insert(node);
            }
        });
        res
    }

    pub fn is_unsafe_macro_call(&self, macro_call: &ast::MacroCall) -> bool {
        let Some(mac) = self.resolve_macro_call(macro_call) else { return false };
        if mac.is_asm_or_global_asm(self.db) {
            return true;
        }

        let Some(sa) = self.analyze(macro_call.syntax()) else { return false };
        let macro_call = self.find_file(macro_call.syntax()).with_value(macro_call);
        match macro_call.map(|it| it.syntax().parent().and_then(ast::MacroExpr::cast)).transpose() {
            Some(it) => sa.is_unsafe_macro_call_expr(self.db, it.as_ref()),
            None => false,
        }
    }

    pub fn resolve_attr_macro_call(&self, item: &ast::Item) -> Option<Macro> {
        let item_in_file = self.wrap_node_infile(item.clone());
        let id = self.with_ctx(|ctx| {
            let macro_call_id = ctx.item_to_macro_call(item_in_file.as_ref())?;
            macro_call_to_macro_id(ctx, macro_call_id)
        })?;
        Some(Macro { id })
    }

    pub fn resolve_path(&self, path: &ast::Path) -> Option<PathResolution> {
        self.resolve_path_with_subst(path).map(|(it, _)| it)
    }

    pub fn resolve_path_per_ns(&self, path: &ast::Path) -> Option<PathResolutionPerNs> {
        self.analyze(path.syntax())?.resolve_hir_path_per_ns(self.db, path)
    }

    pub fn resolve_path_with_subst(
        &self,
        path: &ast::Path,
    ) -> Option<(PathResolution, Option<GenericSubstitution>)> {
        self.analyze(path.syntax())?.resolve_path(self.db, path)
    }

    pub fn resolve_use_type_arg(&self, name: &ast::NameRef) -> Option<TypeParam> {
        self.analyze(name.syntax())?.resolve_use_type_arg(name)
    }

    pub fn resolve_offset_of_field(
        &self,
        name_ref: &ast::NameRef,
    ) -> Option<(Either<Variant, Field>, GenericSubstitution)> {
        self.analyze_no_infer(name_ref.syntax())?.resolve_offset_of_field(self.db, name_ref)
    }

    pub fn resolve_mod_path(
        &self,
        scope: &SyntaxNode,
        path: &ModPath,
    ) -> Option<impl Iterator<Item = ItemInNs>> {
        let analyze = self.analyze(scope)?;
        let items = analyze.resolver.resolve_module_path_in_items(self.db, path);
        Some(items.iter_items().map(|(item, _)| item.into()))
    }

    fn resolve_variant(&self, record_lit: ast::RecordExpr) -> Option<VariantId> {
        self.analyze(record_lit.syntax())?.resolve_variant(record_lit)
    }

    pub fn resolve_bind_pat_to_const(&self, pat: &ast::IdentPat) -> Option<ModuleDef> {
        self.analyze(pat.syntax())?.resolve_bind_pat_to_const(self.db, pat)
    }

    pub fn record_literal_missing_fields(&self, literal: &ast::RecordExpr) -> Vec<(Field, Type)> {
        self.analyze(literal.syntax())
            .and_then(|it| it.record_literal_missing_fields(self.db, literal))
            .unwrap_or_default()
    }

    pub fn record_pattern_missing_fields(&self, pattern: &ast::RecordPat) -> Vec<(Field, Type)> {
        self.analyze(pattern.syntax())
            .and_then(|it| it.record_pattern_missing_fields(self.db, pattern))
            .unwrap_or_default()
    }

    fn with_ctx<F: FnOnce(&mut SourceToDefCtx<'_, '_>) -> T, T>(&self, f: F) -> T {
        let mut ctx = SourceToDefCtx { db: self.db, cache: &mut self.s2d_cache.borrow_mut() };
        f(&mut ctx)
    }

    pub fn to_def<T: ToDef>(&self, src: &T) -> Option<T::Def> {
        let src = self.find_file(src.syntax()).with_value(src);
        T::to_def(self, src)
    }

    fn file_to_module_defs(&self, file: FileId) -> impl Iterator<Item = Module> {
        self.with_ctx(|ctx| ctx.file_to_def(file).to_owned()).into_iter().map(Module::from)
    }

    pub fn scope(&self, node: &SyntaxNode) -> Option<SemanticsScope<'db>> {
        self.analyze_no_infer(node).map(|SourceAnalyzer { file_id, resolver, .. }| SemanticsScope {
            db: self.db,
            file_id,
            resolver,
        })
    }

    pub fn scope_at_offset(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> Option<SemanticsScope<'db>> {
        self.analyze_with_offset_no_infer(node, offset).map(
            |SourceAnalyzer { file_id, resolver, .. }| SemanticsScope {
                db: self.db,
                file_id,
                resolver,
            },
        )
    }

    /// Search for a definition's source and cache its syntax tree
    pub fn source<Def: HasSource>(&self, def: Def) -> Option<InFile<Def::Ast>>
    where
        Def::Ast: AstNode,
    {
        // FIXME: source call should go through the parse cache
        let res = def.source(self.db)?;
        self.cache(find_root(res.value.syntax()), res.file_id);
        Some(res)
    }

    pub fn body_for(&self, node: InFile<&SyntaxNode>) -> Option<DefWithBody> {
        let container = self.with_ctx(|ctx| ctx.find_container(node))?;

        match container {
            ChildContainer::DefWithBodyId(def) => Some(def.into()),
            _ => None,
        }
    }

    /// Returns none if the file of the node is not part of a crate.
    fn analyze(&self, node: &SyntaxNode) -> Option<SourceAnalyzer<'db>> {
        let node = self.find_file(node);
        self.analyze_impl(node, None, true)
    }

    /// Returns none if the file of the node is not part of a crate.
    fn analyze_no_infer(&self, node: &SyntaxNode) -> Option<SourceAnalyzer<'db>> {
        let node = self.find_file(node);
        self.analyze_impl(node, None, false)
    }

    fn analyze_with_offset_no_infer(
        &self,
        node: &SyntaxNode,
        offset: TextSize,
    ) -> Option<SourceAnalyzer<'db>> {
        let node = self.find_file(node);
        self.analyze_impl(node, Some(offset), false)
    }

    fn analyze_impl(
        &self,
        node: InFile<&SyntaxNode>,
        offset: Option<TextSize>,
        // replace this, just make the inference result a `LazyCell`
        infer_body: bool,
    ) -> Option<SourceAnalyzer<'db>> {
        let _p = tracing::info_span!("SemanticsImpl::analyze_impl").entered();

        let container = self.with_ctx(|ctx| ctx.find_container(node))?;

        let resolver = match container {
            ChildContainer::DefWithBodyId(def) => {
                return Some(if infer_body {
                    SourceAnalyzer::new_for_body(self.db, def, node, offset)
                } else {
                    SourceAnalyzer::new_for_body_no_infer(self.db, def, node, offset)
                });
            }
            ChildContainer::VariantId(def) => {
                return Some(SourceAnalyzer::new_variant_body(self.db, def, node, offset));
            }
            ChildContainer::TraitId(it) => {
                return Some(SourceAnalyzer::new_generic_def(self.db, it.into(), node, offset));
            }
            ChildContainer::TraitAliasId(it) => {
                return Some(SourceAnalyzer::new_generic_def(self.db, it.into(), node, offset));
            }
            ChildContainer::ImplId(it) => {
                return Some(SourceAnalyzer::new_generic_def(self.db, it.into(), node, offset));
            }
            ChildContainer::EnumId(it) => {
                return Some(SourceAnalyzer::new_generic_def(self.db, it.into(), node, offset));
            }
            ChildContainer::TypeAliasId(it) => {
                return Some(SourceAnalyzer::new_generic_def(self.db, it.into(), node, offset));
            }
            ChildContainer::GenericDefId(it) => {
                return Some(SourceAnalyzer::new_generic_def(self.db, it, node, offset));
            }
            ChildContainer::ModuleId(it) => it.resolver(self.db),
        };
        Some(SourceAnalyzer::new_for_resolver(resolver, node))
    }

    fn cache(&self, root_node: SyntaxNode, file_id: HirFileId) {
        SourceToDefCache::cache(
            &mut self.s2d_cache.borrow_mut().root_to_file_cache,
            root_node,
            file_id,
        );
    }

    pub fn assert_contains_node(&self, node: &SyntaxNode) {
        self.find_file(node);
    }

    fn lookup(&self, root_node: &SyntaxNode) -> Option<HirFileId> {
        let cache = self.s2d_cache.borrow();
        cache.root_to_file_cache.get(root_node).copied()
    }

    fn wrap_node_infile<N: AstNode>(&self, node: N) -> InFile<N> {
        let InFile { file_id, .. } = self.find_file(node.syntax());
        InFile::new(file_id, node)
    }

    fn wrap_token_infile(&self, token: SyntaxToken) -> InFile<SyntaxToken> {
        let InFile { file_id, .. } = self.find_file(&token.parent().unwrap());
        InFile::new(file_id, token)
    }

    /// Wraps the node in a [`InFile`] with the file id it belongs to.
    fn find_file<'node>(&self, node: &'node SyntaxNode) -> InFile<&'node SyntaxNode> {
        let root_node = find_root(node);
        let file_id = self.lookup(&root_node).unwrap_or_else(|| {
            panic!(
                "\n\nFailed to lookup {:?} in this Semantics.\n\
                 Make sure to only query nodes derived from this instance of Semantics.\n\
                 root node:   {:?}\n\
                 known nodes: {}\n\n",
                node,
                root_node,
                self.s2d_cache
                    .borrow()
                    .root_to_file_cache
                    .keys()
                    .map(|it| format!("{it:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        });
        InFile::new(file_id, node)
    }

    /// Returns `true` if the `node` is inside an `unsafe` context.
    pub fn is_inside_unsafe(&self, expr: &ast::Expr) -> bool {
        let Some(enclosing_item) =
            expr.syntax().ancestors().find_map(Either::<ast::Item, ast::Variant>::cast)
        else {
            return false;
        };

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
                if let Some(ExprOrPatId::ExprId(expr_id)) =
                    source_map.node_expr(InFile { file_id, value: &parent })
                {
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

// FIXME This can't be the best way to do this
fn macro_call_to_macro_id(
    ctx: &mut SourceToDefCtx<'_, '_>,
    macro_call_id: MacroCallId,
) -> Option<MacroId> {
    let db: &dyn ExpandDatabase = ctx.db;
    let loc = db.lookup_intern_macro_call(macro_call_id);

    match loc.def.ast_id() {
        Either::Left(it) => {
            let node = match it.file_id {
                HirFileId::FileId(file_id) => {
                    it.to_ptr(db).to_node(&db.parse(file_id).syntax_node())
                }
                HirFileId::MacroFile(macro_file) => {
                    let expansion_info = ctx.cache.get_or_insert_expansion(ctx.db, macro_file);
                    it.to_ptr(db).to_node(&expansion_info.expanded().value)
                }
            };
            ctx.macro_to_def(InFile::new(it.file_id, &node))
        }
        Either::Right(it) => {
            let node = match it.file_id {
                HirFileId::FileId(file_id) => {
                    it.to_ptr(db).to_node(&db.parse(file_id).syntax_node())
                }
                HirFileId::MacroFile(macro_file) => {
                    let expansion_info = ctx.cache.get_or_insert_expansion(ctx.db, macro_file);
                    it.to_ptr(db).to_node(&expansion_info.expanded().value)
                }
            };
            ctx.proc_macro_to_def(InFile::new(it.file_id, &node))
        }
    }
}

pub trait ToDef: AstNode + Clone {
    type Def;
    fn to_def(sema: &SemanticsImpl<'_>, src: InFile<&Self>) -> Option<Self::Def>;
}

macro_rules! to_def_impls {
    ($(($def:path, $ast:path, $meth:ident)),* ,) => {$(
        impl ToDef for $ast {
            type Def = $def;
            fn to_def(sema: &SemanticsImpl<'_>, src: InFile<&Self>) -> Option<Self::Def> {
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
    (crate::ExternCrateDecl, ast::ExternCrate, extern_crate_to_def),
    (crate::InlineAsmOperand, ast::AsmOperandNamed, asm_operand_to_def),
    (crate::ExternBlock, ast::ExternBlock, extern_block_to_def),
    (MacroCallId, ast::MacroCall, macro_call_to_macro_call),
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
pub struct SemanticsScope<'db> {
    pub db: &'db dyn HirDatabase,
    file_id: HirFileId,
    resolver: Resolver<'db>,
}

impl<'db> SemanticsScope<'db> {
    pub fn module(&self) -> Module {
        Module { id: self.resolver.module() }
    }

    pub fn krate(&self) -> Crate {
        Crate { id: self.resolver.krate() }
    }

    pub fn containing_function(&self) -> Option<Function> {
        self.resolver.body_owner().and_then(|owner| match owner {
            DefWithBodyId::FunctionId(id) => Some(id.into()),
            _ => None,
        })
    }

    pub(crate) fn resolver(&self) -> &Resolver<'db> {
        &self.resolver
    }

    /// Note: `VisibleTraits` should be treated as an opaque type, passed into `Type
    pub fn visible_traits(&self) -> VisibleTraits {
        let resolver = &self.resolver;
        VisibleTraits(resolver.traits_in_scope(self.db))
    }

    /// Calls the passed closure `f` on all names in scope.
    pub fn process_all_names(&self, f: &mut dyn FnMut(Name, ScopeDef)) {
        let scope = self.resolver.names_in_scope(self.db);
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
    pub fn speculative_resolve(&self, ast_path: &ast::Path) -> Option<PathResolution> {
        let mut kind = PathKind::Plain;
        let mut segments = vec![];
        let mut first = true;
        for segment in ast_path.segments() {
            if first {
                first = false;
                if segment.coloncolon_token().is_some() {
                    kind = PathKind::Abs;
                }
            }

            let Some(k) = segment.kind() else { continue };
            match k {
                ast::PathSegmentKind::Name(name_ref) => segments.push(name_ref.as_name()),
                ast::PathSegmentKind::Type { .. } => continue,
                ast::PathSegmentKind::SelfTypeKw => {
                    segments.push(Name::new_symbol_root(sym::Self_))
                }
                ast::PathSegmentKind::SelfKw => kind = PathKind::Super(0),
                ast::PathSegmentKind::SuperKw => match kind {
                    PathKind::Super(s) => kind = PathKind::Super(s + 1),
                    PathKind::Plain => kind = PathKind::Super(1),
                    PathKind::Crate | PathKind::Abs | PathKind::DollarCrate(_) => continue,
                },
                ast::PathSegmentKind::CrateKw => kind = PathKind::Crate,
            }
        }

        resolve_hir_path(
            self.db,
            &self.resolver,
            &Path::BarePath(Interned::new(ModPath::from_segments(kind, segments))),
            name_hygiene(self.db, InFile::new(self.file_id, ast_path.syntax())),
            None,
        )
    }

    pub fn resolve_mod_path(&self, path: &ModPath) -> impl Iterator<Item = ItemInNs> + use<> {
        let items = self.resolver.resolve_module_path_in_items(self.db, path);
        items.iter_items().map(|(item, _)| item.into())
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

    pub fn generic_def(&self) -> Option<crate::GenericDef> {
        self.resolver.generic_def().map(|id| id.into())
    }

    pub fn extern_crates(&self) -> impl Iterator<Item = (Name, Module)> + '_ {
        self.resolver.extern_crates_in_scope().map(|(name, id)| (name, Module { id }))
    }

    pub fn extern_crate_decls(&self) -> impl Iterator<Item = Name> + '_ {
        self.resolver.extern_crate_decls_in_scope(self.db)
    }

    pub fn has_same_self_type(&self, other: &SemanticsScope<'_>) -> bool {
        self.resolver.impl_def() == other.resolver.impl_def()
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

struct RenameConflictsVisitor<'a> {
    db: &'a dyn HirDatabase,
    owner: DefWithBodyId,
    resolver: Resolver<'a>,
    body: &'a Body,
    to_be_renamed: BindingId,
    new_name: Symbol,
    old_name: Symbol,
    conflicts: FxHashSet<BindingId>,
}

impl RenameConflictsVisitor<'_> {
    fn resolve_path(&mut self, node: ExprOrPatId, path: &Path) {
        if let Path::BarePath(path) = path {
            if let Some(name) = path.as_ident() {
                if *name.symbol() == self.new_name {
                    if let Some(conflicting) = self.resolver.rename_will_conflict_with_renamed(
                        self.db,
                        name,
                        path,
                        self.body.expr_or_pat_path_hygiene(node),
                        self.to_be_renamed,
                    ) {
                        self.conflicts.insert(conflicting);
                    }
                } else if *name.symbol() == self.old_name {
                    if let Some(conflicting) =
                        self.resolver.rename_will_conflict_with_another_variable(
                            self.db,
                            name,
                            path,
                            self.body.expr_or_pat_path_hygiene(node),
                            &self.new_name,
                            self.to_be_renamed,
                        )
                    {
                        self.conflicts.insert(conflicting);
                    }
                }
            }
        }
    }

    fn rename_conflicts(&mut self, expr: ExprId) {
        match &self.body[expr] {
            Expr::Path(path) => {
                let guard = self.resolver.update_to_inner_scope(self.db, self.owner, expr);
                self.resolve_path(expr.into(), path);
                self.resolver.reset_to_guard(guard);
            }
            &Expr::Assignment { target, .. } => {
                let guard = self.resolver.update_to_inner_scope(self.db, self.owner, expr);
                self.body.walk_pats(target, &mut |pat| {
                    if let Pat::Path(path) = &self.body[pat] {
                        self.resolve_path(pat.into(), path);
                    }
                });
                self.resolver.reset_to_guard(guard);
            }
            _ => {}
        }

        self.body.walk_child_exprs(expr, |expr| self.rename_conflicts(expr));
    }
}
