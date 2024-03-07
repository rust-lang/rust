//! See `Semantics`.

mod source_to_def;

use std::{
    cell::RefCell,
    fmt, iter, mem,
    ops::{self, ControlFlow, Not},
};

use base_db::{FileId, FileRange};
use either::Either;
use hir_def::{
    hir::Expr,
    lower::LowerCtx,
    nameres::MacroSubNs,
    resolver::{self, HasResolver, Resolver, TypeNs},
    type_ref::Mutability,
    AsMacroCall, DefWithBodyId, FunctionId, MacroId, TraitId, VariantId,
};
use hir_expand::{
    attrs::collect_attrs, db::ExpandDatabase, files::InRealFile, name::AsName, ExpansionInfo,
    InMacroFile, MacroCallId, MacroFileId, MacroFileIdExt,
};
use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{smallvec, SmallVec};
use span::{Span, SyntaxContextId, ROOT_ERASED_FILE_AST_ID};
use stdx::TupleExt;
use syntax::{
    algo::skip_trivia_token,
    ast::{self, HasAttrs as _, HasGenericParams, HasLoopBody, IsString as _},
    match_ast, AstNode, AstToken, Direction, SyntaxKind, SyntaxNode, SyntaxNodePtr, SyntaxToken,
    TextRange, TextSize,
};

use crate::{
    db::HirDatabase,
    semantics::source_to_def::{ChildContainer, SourceToDefCache, SourceToDefCtx},
    source_analyzer::{resolve_hir_path, SourceAnalyzer},
    Access, Adjust, Adjustment, Adt, AutoBorrow, BindingMode, BuiltinAttr, Callable, Const,
    ConstParam, Crate, DeriveHelper, Enum, Field, Function, HasSource, HirFileId, Impl, InFile,
    Label, LifetimeParam, Local, Macro, Module, ModuleDef, Name, OverloadedDeref, Path, ScopeDef,
    Static, Struct, ToolModule, Trait, TraitAlias, TupleField, Type, TypeAlias, TypeParam, Union,
    Variant, VariantDef,
};

pub enum DescendPreference {
    SameText,
    SameKind,
    None,
}

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
    /// Rootnode to HirFileId cache
    cache: RefCell<FxHashMap<SyntaxNode, HirFileId>>,
    // These 2 caches are mainly useful for semantic highlighting as nothing else descends a lot of tokens
    // So we might wanna move them out into something specific for semantic highlighting
    expansion_info_cache: RefCell<FxHashMap<MacroFileId, ExpansionInfo>>,
    /// MacroCall to its expansion's MacroFileId cache
    macro_call_cache: RefCell<FxHashMap<InFile<ast::MacroCall>, MacroFileId>>,
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

impl<'db, DB: HirDatabase> Semantics<'db, DB> {
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

    pub fn file_to_module_def(&self, file: FileId) -> Option<Module> {
        self.imp.file_to_module_defs(file).next()
    }

    pub fn file_to_module_defs(&self, file: FileId) -> impl Iterator<Item = Module> {
        self.imp.file_to_module_defs(file)
    }

    pub fn to_adt_def(&self, a: &ast::Adt) -> Option<Adt> {
        self.imp.to_def(a).map(Adt::from)
    }

    pub fn to_const_def(&self, c: &ast::Const) -> Option<Const> {
        self.imp.to_def(c).map(Const::from)
    }

    pub fn to_enum_def(&self, e: &ast::Enum) -> Option<Enum> {
        self.imp.to_def(e).map(Enum::from)
    }

    pub fn to_enum_variant_def(&self, v: &ast::Variant) -> Option<Variant> {
        self.imp.to_def(v).map(Variant::from)
    }

    pub fn to_fn_def(&self, f: &ast::Fn) -> Option<Function> {
        self.imp.to_def(f).map(Function::from)
    }

    pub fn to_impl_def(&self, i: &ast::Impl) -> Option<Impl> {
        self.imp.to_def(i).map(Impl::from)
    }

    pub fn to_macro_def(&self, m: &ast::Macro) -> Option<Macro> {
        self.imp.to_def(m).map(Macro::from)
    }

    pub fn to_module_def(&self, m: &ast::Module) -> Option<Module> {
        self.imp.to_def(m).map(Module::from)
    }

    pub fn to_static_def(&self, s: &ast::Static) -> Option<Static> {
        self.imp.to_def(s).map(Static::from)
    }

    pub fn to_struct_def(&self, s: &ast::Struct) -> Option<Struct> {
        self.imp.to_def(s).map(Struct::from)
    }

    pub fn to_trait_alias_def(&self, t: &ast::TraitAlias) -> Option<TraitAlias> {
        self.imp.to_def(t).map(TraitAlias::from)
    }

    pub fn to_trait_def(&self, t: &ast::Trait) -> Option<Trait> {
        self.imp.to_def(t).map(Trait::from)
    }

    pub fn to_type_alias_def(&self, t: &ast::TypeAlias) -> Option<TypeAlias> {
        self.imp.to_def(t).map(TypeAlias::from)
    }

    pub fn to_union_def(&self, u: &ast::Union) -> Option<Union> {
        self.imp.to_def(u).map(Union::from)
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

    pub fn parse(&self, file_id: FileId) -> ast::SourceFile {
        let tree = self.db.parse(file_id).tree();
        self.cache(tree.syntax().clone(), file_id.into());
        tree
    }

    pub fn parse_or_expand(&self, file_id: HirFileId) -> SyntaxNode {
        let node = self.db.parse_or_expand(file_id);
        self.cache(node.clone(), file_id);
        node
    }

    pub fn expand(&self, macro_call: &ast::MacroCall) -> Option<SyntaxNode> {
        let sa = self.analyze_no_infer(macro_call.syntax())?;
        let file_id = sa.expand(self.db, InFile::new(sa.file_id, macro_call))?;
        let node = self.parse_or_expand(file_id.into());
        Some(node)
    }

    /// If `item` has an attribute macro attached to it, expands it.
    pub fn expand_attr_macro(&self, item: &ast::Item) -> Option<SyntaxNode> {
        let src = self.wrap_node_infile(item.clone());
        let macro_call_id = self.with_ctx(|ctx| ctx.item_to_macro_call(src))?;
        Some(self.parse_or_expand(macro_call_id.as_file()))
    }

    pub fn expand_derive_as_pseudo_attr_macro(&self, attr: &ast::Attr) -> Option<SyntaxNode> {
        let adt = attr.syntax().parent().and_then(ast::Adt::cast)?;
        let src = self.wrap_node_infile(attr.clone());
        let call_id = self.with_ctx(|ctx| {
            ctx.attr_to_derive_macro_call(src.with_value(&adt), src).map(|(_, it, _)| it)
        })?;
        Some(self.parse_or_expand(call_id.as_file()))
    }

    pub fn resolve_derive_macro(&self, attr: &ast::Attr) -> Option<Vec<Option<Macro>>> {
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

    pub fn expand_derive_macro(&self, attr: &ast::Attr) -> Option<Vec<SyntaxNode>> {
        let res: Vec<_> = self
            .derive_macro_calls(attr)?
            .into_iter()
            .flat_map(|call| {
                let file_id = call?.as_file();
                let node = self.db.parse_or_expand(file_id);
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

    pub fn is_derive_annotated(&self, adt: &ast::Adt) -> bool {
        let file_id = self.find_file(adt.syntax()).file_id;
        let adt = InFile::new(file_id, adt);
        self.with_ctx(|ctx| ctx.has_derives(adt))
    }

    pub fn is_attr_macro_call(&self, item: &ast::Item) -> bool {
        let file_id = self.find_file(item.syntax()).file_id;
        let src = InFile::new(file_id, item.clone());
        self.with_ctx(|ctx| ctx.item_to_macro_call(src).is_some())
    }

    /// Expand the macro call with a different token tree, mapping the `token_to_map` down into the
    /// expansion. `token_to_map` should be a token from the `speculative args` node.
    pub fn speculative_expand(
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
            resolver.resolve_path_as_macro_def(self.db.upcast(), &path, Some(MacroSubNs::Bang))
        })?;
        hir_expand::db::expand_speculative(
            self.db.upcast(),
            macro_call_id,
            speculative_args.syntax(),
            token_to_map,
        )
    }

    /// Expand the macro call with a different item as the input, mapping the `token_to_map` down into the
    /// expansion. `token_to_map` should be a token from the `speculative args` node.
    pub fn speculative_expand_attr_macro(
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

    pub fn speculative_expand_derive_as_pseudo_attr_macro(
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

    pub fn as_format_args_parts(
        &self,
        string: &ast::String,
    ) -> Option<Vec<(TextRange, Option<PathResolution>)>> {
        if let Some(quote) = string.open_quote_text_range() {
            return self
                .descend_into_macros(DescendPreference::SameText, string.syntax().clone())
                .into_iter()
                .find_map(|token| {
                    let string = ast::String::cast(token)?;
                    let literal =
                        string.syntax().parent().filter(|it| it.kind() == SyntaxKind::LITERAL)?;
                    let format_args = ast::FormatArgsExpr::cast(literal.parent()?)?;
                    let source_analyzer = self.analyze_no_infer(format_args.syntax())?;
                    let format_args = self.wrap_node_infile(format_args);
                    let res = source_analyzer
                        .as_format_args_parts(self.db, format_args.as_ref())?
                        .map(|(range, res)| (range + quote.end(), res))
                        .collect();
                    Some(res)
                });
        }
        None
    }

    pub fn check_for_format_args_template(
        &self,
        original_token: SyntaxToken,
        offset: TextSize,
    ) -> Option<(TextRange, Option<PathResolution>)> {
        if let Some(original_string) = ast::String::cast(original_token.clone()) {
            if let Some(quote) = original_string.open_quote_text_range() {
                return self
                    .descend_into_macros(DescendPreference::SameText, original_token)
                    .into_iter()
                    .find_map(|token| {
                        self.resolve_offset_in_format_args(
                            ast::String::cast(token)?,
                            offset.checked_sub(quote.end())?,
                        )
                    })
                    .map(|(range, res)| (range + quote.end(), res));
            }
        }
        None
    }

    fn resolve_offset_in_format_args(
        &self,
        string: ast::String,
        offset: TextSize,
    ) -> Option<(TextRange, Option<PathResolution>)> {
        debug_assert!(offset <= string.syntax().text_range().len());
        let literal = string.syntax().parent().filter(|it| it.kind() == SyntaxKind::LITERAL)?;
        let format_args = ast::FormatArgsExpr::cast(literal.parent()?)?;
        let source_analyzer = &self.analyze_no_infer(format_args.syntax())?;
        let format_args = self.wrap_node_infile(format_args);
        source_analyzer.resolve_offset_in_format_args(self.db, format_args.as_ref(), offset)
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

        if first == last {
            // node is just the token, so descend the token
            self.descend_into_macros_impl(first, &mut |InFile { value, .. }| {
                if let Some(node) = value
                    .parent_ancestors()
                    .take_while(|it| it.text_range() == value.text_range())
                    .find_map(N::cast)
                {
                    res.push(node)
                }
                ControlFlow::Continue(())
            });
        } else {
            // Descend first and last token, then zip them to look for the node they belong to
            let mut scratch: SmallVec<[_; 1]> = smallvec![];
            self.descend_into_macros_impl(first, &mut |token| {
                scratch.push(token);
                ControlFlow::Continue(())
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
                    ControlFlow::Continue(())
                },
            );
        }
        res
    }

    /// Descend the token into its macro call if it is part of one, returning the tokens in the
    /// expansion that it is associated with.
    pub fn descend_into_macros(
        &self,
        mode: DescendPreference,
        token: SyntaxToken,
    ) -> SmallVec<[SyntaxToken; 1]> {
        enum Dp<'t> {
            SameText(&'t str),
            SameKind(SyntaxKind),
            None,
        }
        let fetch_kind = |token: &SyntaxToken| match token.parent() {
            Some(node) => match node.kind() {
                kind @ (SyntaxKind::NAME | SyntaxKind::NAME_REF) => kind,
                _ => token.kind(),
            },
            None => token.kind(),
        };
        let mode = match mode {
            DescendPreference::SameText => Dp::SameText(token.text()),
            DescendPreference::SameKind => Dp::SameKind(fetch_kind(&token)),
            DescendPreference::None => Dp::None,
        };
        let mut res = smallvec![];
        self.descend_into_macros_impl(token.clone(), &mut |InFile { value, .. }| {
            let is_a_match = match mode {
                Dp::SameText(text) => value.text() == text,
                Dp::SameKind(preferred_kind) => {
                    let kind = fetch_kind(&value);
                    kind == preferred_kind
                        // special case for derive macros
                        || (preferred_kind == SyntaxKind::IDENT && kind == SyntaxKind::NAME_REF)
                }
                Dp::None => true,
            };
            if is_a_match {
                res.push(value);
            }
            ControlFlow::Continue(())
        });
        if res.is_empty() {
            res.push(token);
        }
        res
    }

    pub fn descend_into_macros_single(
        &self,
        mode: DescendPreference,
        token: SyntaxToken,
    ) -> SyntaxToken {
        enum Dp<'t> {
            SameText(&'t str),
            SameKind(SyntaxKind),
            None,
        }
        let fetch_kind = |token: &SyntaxToken| match token.parent() {
            Some(node) => match node.kind() {
                kind @ (SyntaxKind::NAME | SyntaxKind::NAME_REF) => kind,
                _ => token.kind(),
            },
            None => token.kind(),
        };
        let mode = match mode {
            DescendPreference::SameText => Dp::SameText(token.text()),
            DescendPreference::SameKind => Dp::SameKind(fetch_kind(&token)),
            DescendPreference::None => Dp::None,
        };
        let mut res = token.clone();
        self.descend_into_macros_impl(token.clone(), &mut |InFile { value, .. }| {
            let is_a_match = match mode {
                Dp::SameText(text) => value.text() == text,
                Dp::SameKind(preferred_kind) => {
                    let kind = fetch_kind(&value);
                    kind == preferred_kind
                        // special case for derive macros
                        || (preferred_kind == SyntaxKind::IDENT && kind == SyntaxKind::NAME_REF)
                }
                Dp::None => true,
            };
            res = value;
            if is_a_match {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        res
    }

    // return:
    // SourceAnalyzer(file_id that original call include!)
    // macro file id
    // token in include! macro mapped from token in params
    // span for the mapped token
    fn is_from_include_file(
        &self,
        token: SyntaxToken,
    ) -> Option<(SourceAnalyzer, HirFileId, SyntaxToken, Span)> {
        let parent = token.parent()?;
        let file_id = self.find_file(&parent).file_id.file_id()?;

        let mut cache = self.expansion_info_cache.borrow_mut();

        // iterate related crates and find all include! invocations that include_file_id matches
        for (invoc, _) in self
            .db
            .relevant_crates(file_id)
            .iter()
            .flat_map(|krate| self.db.include_macro_invoc(*krate))
            .filter(|&(_, include_file_id)| include_file_id == file_id)
        {
            let macro_file = invoc.as_macro_file();
            let expansion_info = cache
                .entry(macro_file)
                .or_insert_with(|| macro_file.expansion_info(self.db.upcast()));

            // Create the source analyzer for the macro call scope
            let Some(sa) = self.analyze_no_infer(&self.parse_or_expand(expansion_info.call_file()))
            else {
                continue;
            };
            {
                let InMacroFile { file_id: macro_file, value } = expansion_info.expanded();
                self.cache(value, macro_file.into());
            }

            // get mapped token in the include! macro file
            let span = span::SpanData {
                range: token.text_range(),
                anchor: span::SpanAnchor { file_id, ast_id: ROOT_ERASED_FILE_AST_ID },
                ctx: SyntaxContextId::ROOT,
            };
            let Some(InMacroFile { file_id, value: mut mapped_tokens }) =
                expansion_info.map_range_down(span)
            else {
                continue;
            };

            // if we find one, then return
            if let Some(t) = mapped_tokens.next() {
                return Some((sa, file_id.into(), t, span));
            }
        }

        None
    }

    fn descend_into_macros_impl(
        &self,
        mut token: SyntaxToken,
        f: &mut dyn FnMut(InFile<SyntaxToken>) -> ControlFlow<()>,
    ) {
        let _p = tracing::span!(tracing::Level::INFO, "descend_into_macros");
        let (sa, span, file_id) =
            match token.parent().and_then(|parent| self.analyze_no_infer(&parent)) {
                Some(sa) => match sa.file_id.file_id() {
                    Some(file_id) => (
                        sa,
                        self.db.real_span_map(file_id).span_for_range(token.text_range()),
                        file_id.into(),
                    ),
                    None => {
                        stdx::never!();
                        return;
                    }
                },
                None => {
                    // if we cannot find a source analyzer for this token, then we try to find out
                    // whether this file is an included file and treat that as the include input
                    let Some((it, macro_file_id, mapped_token, s)) =
                        self.is_from_include_file(token)
                    else {
                        return;
                    };
                    token = mapped_token;
                    (it, s, macro_file_id)
                }
            };

        let mut cache = self.expansion_info_cache.borrow_mut();
        let mut mcache = self.macro_call_cache.borrow_mut();
        let def_map = sa.resolver.def_map();

        let mut stack: Vec<(_, SmallVec<[_; 2]>)> = vec![(file_id, smallvec![token])];

        let mut process_expansion_for_token = |stack: &mut Vec<_>, macro_file| {
            let expansion_info = cache
                .entry(macro_file)
                .or_insert_with(|| macro_file.expansion_info(self.db.upcast()));

            {
                let InMacroFile { file_id, value } = expansion_info.expanded();
                self.cache(value, file_id.into());
            }

            let InMacroFile { file_id, value: mapped_tokens } =
                expansion_info.map_range_down(span)?;
            let mapped_tokens: SmallVec<[_; 2]> = mapped_tokens.collect();

            // if the length changed we have found a mapping for the token
            let res = mapped_tokens.is_empty().not().then_some(());
            // requeue the tokens we got from mapping our current token down
            stack.push((HirFileId::from(file_id), mapped_tokens));
            res
        };

        while let Some((file_id, mut tokens)) = stack.pop() {
            while let Some(token) = tokens.pop() {
                let was_not_remapped = (|| {
                    // First expand into attribute invocations
                    let containing_attribute_macro_call = self.with_ctx(|ctx| {
                        token.parent_ancestors().filter_map(ast::Item::cast).find_map(|item| {
                            // Don't force populate the dyn cache for items that don't have an attribute anyways
                            item.attrs().next()?;
                            Some((
                                ctx.item_to_macro_call(InFile::new(file_id, item.clone()))?,
                                item,
                            ))
                        })
                    });
                    if let Some((call_id, item)) = containing_attribute_macro_call {
                        let file_id = call_id.as_macro_file();
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
                        // remove any other token in this macro input, all their mappings are the
                        // same as this one
                        tokens.retain(|t| !text_range.contains_range(t.text_range()));
                        return process_expansion_for_token(&mut stack, file_id);
                    }

                    // Then check for token trees, that means we are either in a function-like macro or
                    // secondary attribute inputs
                    let tt = token.parent_ancestors().map_while(ast::TokenTree::cast).last()?;
                    let parent = tt.syntax().parent()?;

                    if tt.left_delimiter_token().map_or(false, |it| it == token) {
                        return None;
                    }
                    if tt.right_delimiter_token().map_or(false, |it| it == token) {
                        return None;
                    }

                    if let Some(macro_call) = ast::MacroCall::cast(parent.clone()) {
                        let mcall: hir_expand::files::InFileWrapper<HirFileId, ast::MacroCall> =
                            InFile::new(file_id, macro_call);
                        let file_id = match mcache.get(&mcall) {
                            Some(&it) => it,
                            None => {
                                let it = sa.expand(self.db, mcall.as_ref())?;
                                mcache.insert(mcall, it);
                                it
                            }
                        };
                        let text_range = tt.syntax().text_range();
                        // remove any other token in this macro input, all their mappings are the
                        // same as this one
                        tokens.retain(|t| !text_range.contains_range(t.text_range()));
                        process_expansion_for_token(&mut stack, file_id)
                    } else if let Some(meta) = ast::Meta::cast(parent) {
                        // attribute we failed expansion for earlier, this might be a derive invocation
                        // or derive helper attribute
                        let attr = meta.parent_attr()?;

                        let adt = if let Some(adt) = attr.syntax().parent().and_then(ast::Adt::cast)
                        {
                            // this might be a derive, or a derive helper on an ADT
                            let derive_call = self.with_ctx(|ctx| {
                                // so try downmapping the token into the pseudo derive expansion
                                // see [hir_expand::builtin_attr_macro] for how the pseudo derive expansion works
                                ctx.attr_to_derive_macro_call(
                                    InFile::new(file_id, &adt),
                                    InFile::new(file_id, attr.clone()),
                                )
                                .map(|(_, call_id, _)| call_id)
                            });

                            match derive_call {
                                Some(call_id) => {
                                    // resolved to a derive
                                    let file_id = call_id.as_macro_file();
                                    let text_range = attr.syntax().text_range();
                                    // remove any other token in this macro input, all their mappings are the
                                    // same as this one
                                    tokens.retain(|t| !text_range.contains_range(t.text_range()));
                                    return process_expansion_for_token(&mut stack, file_id);
                                }
                                None => Some(adt),
                            }
                        } else {
                            // Otherwise this could be a derive helper on a variant or field
                            if let Some(field) =
                                attr.syntax().parent().and_then(ast::RecordField::cast)
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
                        if !self.with_ctx(|ctx| ctx.has_derives(InFile::new(file_id, &adt))) {
                            return None;
                        }
                        // Not an attribute, nor a derive, so it's either a builtin or a derive helper
                        // Try to resolve to a derive helper and downmap
                        let attr_name =
                            attr.path().and_then(|it| it.as_single_name_ref())?.as_name();
                        let id = self.db.ast_id_map(file_id).ast_id(&adt);
                        let helpers = def_map.derive_helpers_in_scope(InFile::new(file_id, id))?;
                        let mut res = None;
                        for (.., derive) in
                            helpers.iter().filter(|(helper, ..)| *helper == attr_name)
                        {
                            res = res.or(process_expansion_for_token(
                                &mut stack,
                                derive.as_macro_file(),
                            ));
                        }
                        res
                    } else {
                        None
                    }
                })()
                .is_none();

                if was_not_remapped && f(InFile::new(file_id, token)).is_break() {
                    break;
                }
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
            .map(move |token| self.descend_into_macros(DescendPreference::None, token))
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
        node.original_file_range(self.db.upcast())
    }

    /// Attempts to map the node out of macro expanded files returning the original file range.
    pub fn original_range_opt(&self, node: &SyntaxNode) -> Option<FileRange> {
        let node = self.find_file(node);
        node.original_file_range_opt(self.db.upcast())
            .filter(|(_, ctx)| ctx.is_root())
            .map(TupleExt::head)
    }

    /// Attempts to map the node out of macro expanded files.
    /// This only work for attribute expansions, as other ones do not have nodes as input.
    pub fn original_ast_node<N: AstNode>(&self, node: N) -> Option<N> {
        self.wrap_node_infile(node).original_ast_node_rooted(self.db.upcast()).map(
            |InRealFile { file_id, value }| {
                self.cache(find_root(value.syntax()), file_id.into());
                value
            },
        )
    }

    /// Attempts to map the node out of macro expanded files.
    /// This only work for attribute expansions, as other ones do not have nodes as input.
    pub fn original_syntax_node(&self, node: &SyntaxNode) -> Option<SyntaxNode> {
        let InFile { file_id, .. } = self.find_file(node);
        InFile::new(file_id, node).original_syntax_node(self.db.upcast()).map(
            |InRealFile { file_id, value }| {
                self.cache(find_root(&value), file_id.into());
                value
            },
        )
    }

    pub fn diagnostics_display_range(&self, src: InFile<SyntaxNodePtr>) -> FileRange {
        let root = self.parse_or_expand(src.file_id);
        let node = src.map(|it| it.to_node(&root));
        node.as_ref().original_file_range(self.db.upcast())
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
        let db = self.db.upcast();
        iter::successors(Some(node.cloned()), move |&InFile { file_id, ref value }| {
            match value.parent() {
                Some(parent) => Some(InFile::new(file_id, parent)),
                None => {
                    let call_node = file_id.macro_file()?.call_node(db);
                    // cache the node
                    self.parse_or_expand(call_node.file_id);
                    Some(call_node)
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
        ToDef::to_def(self, src)
    }

    pub fn resolve_label(&self, lifetime: &ast::Lifetime) -> Option<Label> {
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

    pub fn resolve_type(&self, ty: &ast::Type) -> Option<Type> {
        let analyze = self.analyze(ty.syntax())?;
        let ctx = LowerCtx::new(self.db.upcast(), analyze.file_id);
        let ty = hir_ty::TyLoweringContext::new_maybe_unowned(
            self.db,
            &analyze.resolver,
            analyze.resolver.type_owner(),
        )
        .lower_ty(&crate::TypeRef::from_ast(&ctx, ty.clone()));
        Some(Type::new_with_resolver(self.db, &analyze.resolver, ty))
    }

    pub fn resolve_trait(&self, path: &ast::Path) -> Option<Trait> {
        let analyze = self.analyze(path.syntax())?;
        let ctx = LowerCtx::new(self.db.upcast(), analyze.file_id);
        let hir_path = Path::from_src(&ctx, path.clone())?;
        match analyze.resolver.resolve_path_in_type_ns_fully(self.db.upcast(), &hir_path)? {
            TypeNs::TraitId(id) => Some(Trait { id }),
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
    ) -> Option<Either<Function, Field>> {
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

    pub fn resolve_method_call_as_callable(&self, call: &ast::MethodCallExpr) -> Option<Callable> {
        self.analyze(call.syntax())?.resolve_method_call_as_callable(self.db, call)
    }

    pub fn resolve_field(&self, field: &ast::FieldExpr) -> Option<Either<Field, TupleField>> {
        self.analyze(field.syntax())?.resolve_field(self.db, field)
    }

    pub fn resolve_field_fallback(
        &self,
        field: &ast::FieldExpr,
    ) -> Option<Either<Either<Field, TupleField>, Function>> {
        self.analyze(field.syntax())?.resolve_field_fallback(self.db, field)
    }

    pub fn resolve_record_field(
        &self,
        field: &ast::RecordExprField,
    ) -> Option<(Field, Option<Local>, Type)> {
        self.analyze(field.syntax())?.resolve_record_field(self.db, field)
    }

    pub fn resolve_record_pat_field(&self, field: &ast::RecordPatField) -> Option<(Field, Type)> {
        self.analyze(field.syntax())?.resolve_record_pat_field(self.db, field)
    }

    pub fn resolve_macro_call(&self, macro_call: &ast::MacroCall) -> Option<Macro> {
        let sa = self.analyze(macro_call.syntax())?;
        let macro_call = self.find_file(macro_call.syntax()).with_value(macro_call);
        sa.resolve_macro_call(self.db, macro_call)
    }

    pub fn is_unsafe_macro_call(&self, macro_call: &ast::MacroCall) -> bool {
        let sa = match self.analyze(macro_call.syntax()) {
            Some(it) => it,
            None => return false,
        };
        let macro_call = self.find_file(macro_call.syntax()).with_value(macro_call);
        sa.is_unsafe_macro_call(self.db, macro_call)
    }

    pub fn resolve_attr_macro_call(&self, item: &ast::Item) -> Option<Macro> {
        let item_in_file = self.wrap_node_infile(item.clone());
        let id = self.with_ctx(|ctx| {
            let macro_call_id = ctx.item_to_macro_call(item_in_file)?;
            macro_call_to_macro_id(ctx, self.db.upcast(), macro_call_id)
        })?;
        Some(Macro { id })
    }

    pub fn resolve_path(&self, path: &ast::Path) -> Option<PathResolution> {
        self.analyze(path.syntax())?.resolve_path(self.db, path)
    }

    fn resolve_variant(&self, record_lit: ast::RecordExpr) -> Option<VariantId> {
        self.analyze(record_lit.syntax())?.resolve_variant(self.db, record_lit)
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
        let mut cache = self.s2d_cache.borrow_mut();
        let mut ctx = SourceToDefCtx { db: self.db, dynmap_cache: &mut cache };
        f(&mut ctx)
    }

    pub fn to_def<T: ToDef>(&self, src: &T) -> Option<T::Def> {
        let src = self.find_file(src.syntax()).with_value(src).cloned();
        T::to_def(self, src)
    }

    fn file_to_module_defs(&self, file: FileId) -> impl Iterator<Item = Module> {
        self.with_ctx(|ctx| ctx.file_to_def(file)).into_iter().map(Module::from)
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
        let _p = tracing::span!(tracing::Level::INFO, "Semantics::analyze_impl");
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
        assert!(prev.is_none() || prev == Some(file_id))
    }

    pub fn assert_contains_node(&self, node: &SyntaxNode) {
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

    pub fn is_unsafe_method_call(&self, method_call_expr: &ast::MethodCallExpr) -> bool {
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

                let func = self.resolve_method_call(method_call_expr)?;
                let res = match func.self_param(self.db)?.access(self.db) {
                    Access::Shared | Access::Exclusive => true,
                    Access::Owned => false,
                };
                Some(res)
            })
            .unwrap_or(false)
    }

    pub fn is_unsafe_ref_expr(&self, ref_expr: &ast::RefExpr) -> bool {
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

    pub fn is_unsafe_ident_pat(&self, ident_pat: &ast::IdentPat) -> bool {
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
    (crate::ExternCrateDecl, ast::ExternCrate, extern_crate_to_def),
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

impl SemanticsScope<'_> {
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

    /// Calls the passed closure `f` on all names in scope.
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
        let ctx = LowerCtx::new(self.db.upcast(), self.file_id);
        let path = Path::from_src(&ctx, path.clone())?;
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

    pub fn extern_crates(&self) -> impl Iterator<Item = (Name, Module)> + '_ {
        self.resolver.extern_crates_in_scope().map(|(name, id)| (name, Module { id }))
    }

    pub fn extern_crate_decls(&self) -> impl Iterator<Item = Name> + '_ {
        self.resolver.extern_crate_decls_in_scope(self.db.upcast())
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
