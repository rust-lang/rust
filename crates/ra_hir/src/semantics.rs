//! See `Semantics`.

mod source_to_def;

use std::{cell::RefCell, fmt, iter::successors};

use hir_def::{
    resolver::{self, HasResolver, Resolver},
    AsMacroCall, TraitId,
};
use hir_expand::ExpansionInfo;
use ra_db::{FileId, FileRange};
use ra_prof::profile;
use ra_syntax::{
    algo::{find_node_at_offset, skip_trivia_token},
    ast, AstNode, Direction, SyntaxNode, SyntaxToken, TextRange, TextUnit,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    db::HirDatabase,
    semantics::source_to_def::{ChildContainer, SourceToDefCache, SourceToDefCtx},
    source_analyzer::{resolve_hir_path, SourceAnalyzer},
    AssocItem, Function, HirFileId, ImplDef, InFile, Local, MacroDef, Module, ModuleDef, Name,
    Origin, Path, ScopeDef, StructField, Trait, Type, TypeParam, VariantDef,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathResolution {
    /// An item
    Def(ModuleDef),
    /// A local binding (only value namespace)
    Local(Local),
    /// A generic parameter
    TypeParam(TypeParam),
    SelfType(ImplDef),
    Macro(MacroDef),
    AssocItem(AssocItem),
}

/// Primary API to get semantic information, like types, from syntax trees.
pub struct Semantics<'db, DB> {
    pub db: &'db DB,
    s2d_cache: RefCell<SourceToDefCache>,
    cache: RefCell<FxHashMap<SyntaxNode, HirFileId>>,
}

impl<DB> fmt::Debug for Semantics<'_, DB> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Semantics {{ ... }}")
    }
}

impl<'db, DB: HirDatabase> Semantics<'db, DB> {
    pub fn new(db: &DB) -> Semantics<DB> {
        Semantics { db, s2d_cache: Default::default(), cache: Default::default() }
    }

    pub fn parse(&self, file_id: FileId) -> ast::SourceFile {
        let tree = self.db.parse(file_id).tree();
        self.cache(tree.syntax().clone(), file_id.into());
        tree
    }

    pub fn expand(&self, macro_call: &ast::MacroCall) -> Option<SyntaxNode> {
        let macro_call = self.find_file(macro_call.syntax().clone()).with_value(macro_call);
        let sa = self.analyze2(macro_call.map(|it| it.syntax()), None);
        let file_id = sa.expand(self.db, macro_call)?;
        let node = self.db.parse_or_expand(file_id)?;
        self.cache(node.clone(), file_id);
        Some(node)
    }

    pub fn expand_hypothetical(
        &self,
        actual_macro_call: &ast::MacroCall,
        hypothetical_args: &ast::TokenTree,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, SyntaxToken)> {
        let macro_call =
            self.find_file(actual_macro_call.syntax().clone()).with_value(actual_macro_call);
        let sa = self.analyze2(macro_call.map(|it| it.syntax()), None);
        let macro_call_id = macro_call
            .as_call_id(self.db, |path| sa.resolver.resolve_path_as_macro(self.db, &path))?;
        hir_expand::db::expand_hypothetical(self.db, macro_call_id, hypothetical_args, token_to_map)
    }

    pub fn descend_into_macros(&self, token: SyntaxToken) -> SyntaxToken {
        let parent = token.parent();
        let parent = self.find_file(parent);
        let sa = self.analyze2(parent.as_ref(), None);

        let token = successors(Some(parent.with_value(token)), |token| {
            let macro_call = token.value.ancestors().find_map(ast::MacroCall::cast)?;
            let tt = macro_call.token_tree()?;
            if !token.value.text_range().is_subrange(&tt.syntax().text_range()) {
                return None;
            }
            let file_id = sa.expand(self.db, token.with_value(&macro_call))?;
            let token = file_id.expansion_info(self.db)?.map_token_down(token.as_ref())?;

            self.cache(find_root(&token.value.parent()), token.file_id);

            Some(token)
        })
        .last()
        .unwrap();

        token.value
    }

    pub fn descend_node_at_offset<N: ast::AstNode>(
        &self,
        node: &SyntaxNode,
        offset: TextUnit,
    ) -> Option<N> {
        // Handle macro token cases
        node.token_at_offset(offset)
            .map(|token| self.descend_into_macros(token))
            .find_map(|it| self.ancestors_with_macros(it.parent()).find_map(N::cast))
    }

    pub fn original_range(&self, node: &SyntaxNode) -> FileRange {
        let node = self.find_file(node.clone());
        original_range(self.db, node.as_ref())
    }

    pub fn ancestors_with_macros(&self, node: SyntaxNode) -> impl Iterator<Item = SyntaxNode> + '_ {
        let node = self.find_file(node);
        node.ancestors_with_macros(self.db).map(|it| it.value)
    }

    pub fn ancestors_at_offset_with_macros(
        &self,
        node: &SyntaxNode,
        offset: TextUnit,
    ) -> impl Iterator<Item = SyntaxNode> + '_ {
        use itertools::Itertools;
        node.token_at_offset(offset)
            .map(|token| self.ancestors_with_macros(token.parent()))
            .kmerge_by(|node1, node2| node1.text_range().len() < node2.text_range().len())
    }

    /// Find a AstNode by offset inside SyntaxNode, if it is inside *Macrofile*,
    /// search up until it is of the target AstNode type
    pub fn find_node_at_offset_with_macros<N: AstNode>(
        &self,
        node: &SyntaxNode,
        offset: TextUnit,
    ) -> Option<N> {
        self.ancestors_at_offset_with_macros(node, offset).find_map(N::cast)
    }

    /// Find a AstNode by offset inside SyntaxNode, if it is inside *MacroCall*,
    /// descend it and find again
    pub fn find_node_at_offset_with_descend<N: AstNode>(
        &self,
        node: &SyntaxNode,
        offset: TextUnit,
    ) -> Option<N> {
        if let Some(it) = find_node_at_offset(&node, offset) {
            return Some(it);
        }
        self.descend_node_at_offset(&node, offset)
    }

    pub fn type_of_expr(&self, expr: &ast::Expr) -> Option<Type> {
        self.analyze(expr.syntax()).type_of(self.db, &expr)
    }

    pub fn type_of_pat(&self, pat: &ast::Pat) -> Option<Type> {
        self.analyze(pat.syntax()).type_of_pat(self.db, &pat)
    }

    pub fn resolve_method_call(&self, call: &ast::MethodCallExpr) -> Option<Function> {
        self.analyze(call.syntax()).resolve_method_call(self.db, call)
    }

    pub fn resolve_field(&self, field: &ast::FieldExpr) -> Option<StructField> {
        self.analyze(field.syntax()).resolve_field(self.db, field)
    }

    pub fn resolve_record_field(
        &self,
        field: &ast::RecordField,
    ) -> Option<(StructField, Option<Local>)> {
        self.analyze(field.syntax()).resolve_record_field(self.db, field)
    }

    pub fn resolve_record_literal(&self, record_lit: &ast::RecordLit) -> Option<VariantDef> {
        self.analyze(record_lit.syntax()).resolve_record_literal(self.db, record_lit)
    }

    pub fn resolve_record_pattern(&self, record_pat: &ast::RecordPat) -> Option<VariantDef> {
        self.analyze(record_pat.syntax()).resolve_record_pattern(record_pat)
    }

    pub fn resolve_macro_call(&self, macro_call: &ast::MacroCall) -> Option<MacroDef> {
        let sa = self.analyze(macro_call.syntax());
        let macro_call = self.find_file(macro_call.syntax().clone()).with_value(macro_call);
        sa.resolve_macro_call(self.db, macro_call)
    }

    pub fn resolve_path(&self, path: &ast::Path) -> Option<PathResolution> {
        self.analyze(path.syntax()).resolve_path(self.db, path)
    }

    pub fn resolve_bind_pat_to_const(&self, pat: &ast::BindPat) -> Option<ModuleDef> {
        self.analyze(pat.syntax()).resolve_bind_pat_to_const(self.db, pat)
    }

    // FIXME: use this instead?
    // pub fn resolve_name_ref(&self, name_ref: &ast::NameRef) -> Option<???>;

    pub fn to_def<T: ToDef>(&self, src: &T) -> Option<T::Def> {
        let src = self.find_file(src.syntax().clone()).with_value(src).cloned();
        T::to_def(self, src)
    }

    fn with_ctx<F: FnOnce(&mut SourceToDefCtx) -> T, T>(&self, f: F) -> T {
        let mut cache = self.s2d_cache.borrow_mut();
        let mut ctx = SourceToDefCtx { db: self.db, cache: &mut *cache };
        f(&mut ctx)
    }

    pub fn to_module_def(&self, file: FileId) -> Option<Module> {
        self.with_ctx(|ctx| ctx.file_to_def(file)).map(Module::from)
    }

    pub fn scope(&self, node: &SyntaxNode) -> SemanticsScope<'db, DB> {
        let node = self.find_file(node.clone());
        let resolver = self.analyze2(node.as_ref(), None).resolver;
        SemanticsScope { db: self.db, resolver }
    }

    pub fn scope_at_offset(&self, node: &SyntaxNode, offset: TextUnit) -> SemanticsScope<'db, DB> {
        let node = self.find_file(node.clone());
        let resolver = self.analyze2(node.as_ref(), Some(offset)).resolver;
        SemanticsScope { db: self.db, resolver }
    }

    pub fn scope_for_def(&self, def: Trait) -> SemanticsScope<'db, DB> {
        let resolver = def.id.resolver(self.db);
        SemanticsScope { db: self.db, resolver }
    }

    fn analyze(&self, node: &SyntaxNode) -> SourceAnalyzer {
        let src = self.find_file(node.clone());
        self.analyze2(src.as_ref(), None)
    }

    fn analyze2(&self, src: InFile<&SyntaxNode>, offset: Option<TextUnit>) -> SourceAnalyzer {
        let _p = profile("Semantics::analyze2");

        let container = match self.with_ctx(|ctx| ctx.find_container(src)) {
            Some(it) => it,
            None => return SourceAnalyzer::new_for_resolver(Resolver::default(), src),
        };

        let resolver = match container {
            ChildContainer::DefWithBodyId(def) => {
                return SourceAnalyzer::new_for_body(self.db, def, src, offset)
            }
            ChildContainer::TraitId(it) => it.resolver(self.db),
            ChildContainer::ImplId(it) => it.resolver(self.db),
            ChildContainer::ModuleId(it) => it.resolver(self.db),
            ChildContainer::EnumId(it) => it.resolver(self.db),
            ChildContainer::VariantId(it) => it.resolver(self.db),
            ChildContainer::GenericDefId(it) => it.resolver(self.db),
        };
        SourceAnalyzer::new_for_resolver(resolver, src)
    }

    fn cache(&self, root_node: SyntaxNode, file_id: HirFileId) {
        assert!(root_node.parent().is_none());
        let mut cache = self.cache.borrow_mut();
        let prev = cache.insert(root_node, file_id);
        assert!(prev == None || prev == Some(file_id))
    }

    pub fn assert_contains_node(&self, node: &SyntaxNode) {
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
}

pub trait ToDef: AstNode + Clone {
    type Def;

    fn to_def<DB: HirDatabase>(sema: &Semantics<DB>, src: InFile<Self>) -> Option<Self::Def>;
}

macro_rules! to_def_impls {
    ($(($def:path, $ast:path, $meth:ident)),* ,) => {$(
        impl ToDef for $ast {
            type Def = $def;
            fn to_def<DB: HirDatabase>(sema: &Semantics<DB>, src: InFile<Self>) -> Option<Self::Def> {
                sema.with_ctx(|ctx| ctx.$meth(src)).map(<$def>::from)
            }
        }
    )*}
}

to_def_impls![
    (crate::Module, ast::Module, module_to_def),
    (crate::Struct, ast::StructDef, struct_to_def),
    (crate::Enum, ast::EnumDef, enum_to_def),
    (crate::Union, ast::UnionDef, union_to_def),
    (crate::Trait, ast::TraitDef, trait_to_def),
    (crate::ImplDef, ast::ImplDef, impl_to_def),
    (crate::TypeAlias, ast::TypeAliasDef, type_alias_to_def),
    (crate::Const, ast::ConstDef, const_to_def),
    (crate::Static, ast::StaticDef, static_to_def),
    (crate::Function, ast::FnDef, fn_to_def),
    (crate::StructField, ast::RecordFieldDef, record_field_to_def),
    (crate::StructField, ast::TupleFieldDef, tuple_field_to_def),
    (crate::EnumVariant, ast::EnumVariant, enum_variant_to_def),
    (crate::TypeParam, ast::TypeParam, type_param_to_def),
    (crate::MacroDef, ast::MacroCall, macro_call_to_def), // this one is dubious, not all calls are macros
    (crate::Local, ast::BindPat, bind_pat_to_def),
];

fn find_root(node: &SyntaxNode) -> SyntaxNode {
    node.ancestors().last().unwrap()
}

pub struct SemanticsScope<'a, DB> {
    pub db: &'a DB,
    resolver: Resolver,
}

impl<'a, DB: HirDatabase> SemanticsScope<'a, DB> {
    pub fn module(&self) -> Option<Module> {
        Some(Module { id: self.resolver.module()? })
    }

    /// Note: `FxHashSet<TraitId>` should be treated as an opaque type, passed into `Type
    // FIXME: rename to visible_traits to not repeat scope?
    pub fn traits_in_scope(&self) -> FxHashSet<TraitId> {
        let resolver = &self.resolver;
        resolver.traits_in_scope(self.db)
    }

    pub fn process_all_names(&self, f: &mut dyn FnMut(Name, ScopeDef)) {
        let resolver = &self.resolver;

        resolver.process_all_names(self.db, &mut |name, def| {
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
                resolver::ScopeDef::GenericParam(id) => ScopeDef::GenericParam(TypeParam { id }),
                resolver::ScopeDef::Local(pat_id) => {
                    let parent = resolver.body_owner().unwrap().into();
                    ScopeDef::Local(Local { parent, pat_id })
                }
            };
            f(name, def)
        })
    }

    pub fn resolve_hir_path(&self, path: &Path) -> Option<PathResolution> {
        resolve_hir_path(self.db, &self.resolver, path)
    }
}

// FIXME: Change `HasSource` trait to work with `Semantics` and remove this?
pub fn original_range(db: &dyn HirDatabase, node: InFile<&SyntaxNode>) -> FileRange {
    if let Some(range) = original_range_opt(db, node) {
        let original_file = range.file_id.original_file(db.upcast());
        if range.file_id == original_file.into() {
            return FileRange { file_id: original_file, range: range.value };
        }

        log::error!("Fail to mapping up more for {:?}", range);
        return FileRange { file_id: range.file_id.original_file(db.upcast()), range: range.value };
    }

    // Fall back to whole macro call
    if let Some(expansion) = node.file_id.expansion_info(db.upcast()) {
        if let Some(call_node) = expansion.call_node() {
            return FileRange {
                file_id: call_node.file_id.original_file(db.upcast()),
                range: call_node.value.text_range(),
            };
        }
    }

    FileRange { file_id: node.file_id.original_file(db.upcast()), range: node.value.text_range() }
}

fn original_range_opt(
    db: &dyn HirDatabase,
    node: InFile<&SyntaxNode>,
) -> Option<InFile<TextRange>> {
    let expansion = node.file_id.expansion_info(db.upcast())?;

    // the input node has only one token ?
    let single = skip_trivia_token(node.value.first_token()?, Direction::Next)?
        == skip_trivia_token(node.value.last_token()?, Direction::Prev)?;

    Some(node.value.descendants().find_map(|it| {
        let first = skip_trivia_token(it.first_token()?, Direction::Next)?;
        let first = ascend_call_token(db, &expansion, node.with_value(first))?;

        let last = skip_trivia_token(it.last_token()?, Direction::Prev)?;
        let last = ascend_call_token(db, &expansion, node.with_value(last))?;

        if (!single && first == last) || (first.file_id != last.file_id) {
            return None;
        }

        Some(first.with_value(first.value.text_range().extend_to(&last.value.text_range())))
    })?)
}

fn ascend_call_token(
    db: &dyn HirDatabase,
    expansion: &ExpansionInfo,
    token: InFile<SyntaxToken>,
) -> Option<InFile<SyntaxToken>> {
    let (mapped, origin) = expansion.map_token_up(token.as_ref())?;
    if origin != Origin::Call {
        return None;
    }
    if let Some(info) = mapped.file_id.expansion_info(db.upcast()) {
        return ascend_call_token(db, &info, mapped);
    }
    Some(mapped)
}
