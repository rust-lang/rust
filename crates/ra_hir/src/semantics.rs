//! See `Semantics`.

use std::{cell::RefCell, fmt, iter::successors};

use hir_def::{
    resolver::{self, HasResolver, Resolver},
    TraitId,
};
use ra_db::{FileId, FileRange};
use ra_syntax::{ast, AstNode, SyntaxNode, SyntaxToken, TextRange, TextUnit};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    db::HirDatabase,
    source_analyzer::{resolve_hir_path, ReferenceDescriptor, SourceAnalyzer},
    source_binder::{ChildContainer, SourceBinder, ToDef},
    Function, HirFileId, InFile, Local, MacroDef, Module, Name, Origin, Path, PathResolution,
    ScopeDef, StructField, Trait, Type, TypeParam, VariantDef,
};
use ra_prof::profile;

/// Primary API to get semantic information, like types, from syntax trees.
pub struct Semantics<'db, DB> {
    pub db: &'db DB,
    pub(crate) sb: RefCell<SourceBinder>,
    cache: RefCell<FxHashMap<SyntaxNode, HirFileId>>,
}

impl<DB> fmt::Debug for Semantics<'_, DB> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Semantics {{ ... }}")
    }
}

impl<'db, DB: HirDatabase> Semantics<'db, DB> {
    pub fn new(db: &DB) -> Semantics<DB> {
        let sb = RefCell::new(SourceBinder::new());
        Semantics { db, sb, cache: RefCell::default() }
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

    pub fn original_range(&self, node: &SyntaxNode) -> FileRange {
        let node = self.find_file(node.clone());
        original_range(self.db, node.as_ref())
    }

    pub fn ancestors_with_macros(&self, node: SyntaxNode) -> impl Iterator<Item = SyntaxNode> + '_ {
        let node = self.find_file(node);
        node.ancestors_with_macros(self.db).map(|it| it.value)
    }

    pub fn type_of_expr(&self, expr: &ast::Expr) -> Option<Type> {
        self.analyze(expr.syntax()).type_of(self.db, &expr)
    }

    pub fn type_of_pat(&self, pat: &ast::Pat) -> Option<Type> {
        self.analyze(pat.syntax()).type_of_pat(self.db, &pat)
    }

    pub fn resolve_method_call(&self, call: &ast::MethodCallExpr) -> Option<Function> {
        self.analyze(call.syntax()).resolve_method_call(call)
    }

    pub fn resolve_field(&self, field: &ast::FieldExpr) -> Option<StructField> {
        self.analyze(field.syntax()).resolve_field(field)
    }

    pub fn resolve_record_field(&self, field: &ast::RecordField) -> Option<StructField> {
        self.analyze(field.syntax()).resolve_record_field(field)
    }

    pub fn resolve_record_literal(&self, record_lit: &ast::RecordLit) -> Option<VariantDef> {
        self.analyze(record_lit.syntax()).resolve_record_literal(record_lit)
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

    // FIXME: use this instead?
    // pub fn resolve_name_ref(&self, name_ref: &ast::NameRef) -> Option<???>;

    pub fn to_def<T: ToDef + Clone>(&self, src: &T) -> Option<T::Def> {
        let src = self.find_file(src.syntax().clone()).with_value(src.clone());
        let mut sb = self.sb.borrow_mut();
        T::to_def(self.db, &mut sb, src)
    }

    pub fn to_module_def(&self, file: FileId) -> Option<Module> {
        let mut sb = self.sb.borrow_mut();
        sb.to_module_def(self.db, file)
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

    // FIXME: we only use this in `inline_local_variable` assist, ideally, we
    // should switch to general reference search infra there.
    pub fn find_all_refs(&self, pat: &ast::BindPat) -> Vec<ReferenceDescriptor> {
        self.analyze(pat.syntax()).find_all_refs(pat)
    }

    fn analyze(&self, node: &SyntaxNode) -> SourceAnalyzer {
        let src = self.find_file(node.clone());
        self.analyze2(src.as_ref(), None)
    }

    fn analyze2(&self, src: InFile<&SyntaxNode>, offset: Option<TextUnit>) -> SourceAnalyzer {
        let _p = profile("Semantics::analyze2");

        let container = match self.sb.borrow_mut().find_container(self.db, src) {
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
                resolver::ScopeDef::PerNs(it) => it.into(),
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
pub fn original_range(db: &impl HirDatabase, node: InFile<&SyntaxNode>) -> FileRange {
    if let Some((range, Origin::Call)) = original_range_and_origin(db, node) {
        return range;
    }

    if let Some(expansion) = node.file_id.expansion_info(db) {
        if let Some(call_node) = expansion.call_node() {
            return FileRange {
                file_id: call_node.file_id.original_file(db),
                range: call_node.value.text_range(),
            };
        }
    }

    FileRange { file_id: node.file_id.original_file(db), range: node.value.text_range() }
}

fn original_range_and_origin(
    db: &impl HirDatabase,
    node: InFile<&SyntaxNode>,
) -> Option<(FileRange, Origin)> {
    let expansion = node.file_id.expansion_info(db)?;

    // the input node has only one token ?
    let single = node.value.first_token()? == node.value.last_token()?;

    // FIXME: We should handle recurside macro expansions
    let (range, origin) = node.value.descendants().find_map(|it| {
        let first = it.first_token()?;
        let last = it.last_token()?;

        if !single && first == last {
            return None;
        }

        // Try to map first and last tokens of node, and, if success, return the union range of mapped tokens
        let (first, first_origin) = expansion.map_token_up(node.with_value(&first))?;
        let (last, last_origin) = expansion.map_token_up(node.with_value(&last))?;

        if first.file_id != last.file_id || first_origin != last_origin {
            return None;
        }

        // FIXME: Add union method in TextRange
        Some((
            first.with_value(union_range(first.value.text_range(), last.value.text_range())),
            first_origin,
        ))
    })?;

    return Some((
        FileRange { file_id: range.file_id.original_file(db), range: range.value },
        origin,
    ));

    fn union_range(a: TextRange, b: TextRange) -> TextRange {
        let start = a.start().min(b.start());
        let end = a.end().max(b.end());
        TextRange::from_to(start, end)
    }
}
