//! Implementation of find-usages functionality.
//!
//! It is based on the standard ide trick: first, we run a fast text search to
//! get a super-set of matches. Then, we we confirm each match using precise
//! name resolution.

use std::{convert::TryInto, mem};

use base_db::{FileId, FileRange, SourceDatabase, SourceDatabaseExt};
use hir::{
    DefWithBody, HasAttrs, HasSource, InFile, ModuleDef, ModuleSource, Semantics, Visibility,
};
use once_cell::unsync::Lazy;
use rustc_hash::FxHashMap;
use syntax::{ast, match_ast, AstNode, TextRange, TextSize};

use crate::{
    defs::{Definition, NameClass, NameRefClass},
    RootDatabase,
};

#[derive(Debug, Default, Clone)]
pub struct UsageSearchResult {
    pub references: FxHashMap<FileId, Vec<FileReference>>,
}

impl UsageSearchResult {
    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
    }

    pub fn len(&self) -> usize {
        self.references.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&FileId, &Vec<FileReference>)> + '_ {
        self.references.iter()
    }

    pub fn file_ranges(&self) -> impl Iterator<Item = FileRange> + '_ {
        self.references.iter().flat_map(|(&file_id, refs)| {
            refs.iter().map(move |&FileReference { range, .. }| FileRange { file_id, range })
        })
    }
}

impl IntoIterator for UsageSearchResult {
    type Item = (FileId, Vec<FileReference>);
    type IntoIter = <FxHashMap<FileId, Vec<FileReference>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.references.into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct FileReference {
    pub range: TextRange,
    pub name: ast::NameLike,
    pub access: Option<ReferenceAccess>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ReferenceAccess {
    Read,
    Write,
}

/// Generally, `search_scope` returns files that might contain references for the element.
/// For `pub(crate)` things it's a crate, for `pub` things it's a crate and dependant crates.
/// In some cases, the location of the references is known to within a `TextRange`,
/// e.g. for things like local variables.
pub struct SearchScope {
    entries: FxHashMap<FileId, Option<TextRange>>,
}

impl SearchScope {
    fn new(entries: FxHashMap<FileId, Option<TextRange>>) -> SearchScope {
        SearchScope { entries }
    }

    fn crate_graph(db: &RootDatabase) -> SearchScope {
        let mut entries = FxHashMap::default();

        let graph = db.crate_graph();
        for krate in graph.iter() {
            let root_file = graph[krate].root_file_id;
            let source_root_id = db.file_source_root(root_file);
            let source_root = db.source_root(source_root_id);
            entries.extend(source_root.iter().map(|id| (id, None)));
        }
        SearchScope { entries }
    }

    fn reverse_dependencies(db: &RootDatabase, of: hir::Crate) -> SearchScope {
        let mut entries = FxHashMap::default();
        for rev_dep in of.transitive_reverse_dependencies(db) {
            let root_file = rev_dep.root_file(db);
            let source_root_id = db.file_source_root(root_file);
            let source_root = db.source_root(source_root_id);
            entries.extend(source_root.iter().map(|id| (id, None)));
        }
        SearchScope { entries }
    }

    fn krate(db: &RootDatabase, of: hir::Crate) -> SearchScope {
        let root_file = of.root_file(db);
        let source_root_id = db.file_source_root(root_file);
        let source_root = db.source_root(source_root_id);
        SearchScope {
            entries: source_root.iter().map(|id| (id, None)).collect::<FxHashMap<_, _>>(),
        }
    }

    fn module(db: &RootDatabase, module: hir::Module) -> SearchScope {
        let mut entries = FxHashMap::default();

        let mut to_visit = vec![module];
        let mut is_first = true;
        while let Some(module) = to_visit.pop() {
            let src = module.definition_source(db);
            let file_id = src.file_id.original_file(db);
            match src.value {
                ModuleSource::Module(m) => {
                    if is_first {
                        let range = Some(m.syntax().text_range());
                        entries.insert(file_id, range);
                    } else {
                        // We have already added the enclosing file to the search scope,
                        // so do nothing.
                    }
                }
                ModuleSource::BlockExpr(b) => {
                    if is_first {
                        let range = Some(b.syntax().text_range());
                        entries.insert(file_id, range);
                    } else {
                        // We have already added the enclosing file to the search scope,
                        // so do nothing.
                    }
                }
                ModuleSource::SourceFile(_) => {
                    entries.insert(file_id, None);
                }
            };
            is_first = false;
            to_visit.extend(module.children(db));
        }
        SearchScope { entries }
    }

    pub fn empty() -> SearchScope {
        SearchScope::new(FxHashMap::default())
    }

    pub fn single_file(file: FileId) -> SearchScope {
        SearchScope::new(std::iter::once((file, None)).collect())
    }

    pub fn file_range(range: FileRange) -> SearchScope {
        SearchScope::new(std::iter::once((range.file_id, Some(range.range))).collect())
    }

    pub fn files(files: &[FileId]) -> SearchScope {
        SearchScope::new(files.iter().map(|f| (*f, None)).collect())
    }

    pub fn intersection(&self, other: &SearchScope) -> SearchScope {
        let (mut small, mut large) = (&self.entries, &other.entries);
        if small.len() > large.len() {
            mem::swap(&mut small, &mut large)
        }

        let res = small
            .iter()
            .filter_map(|(file_id, r1)| {
                let r2 = large.get(file_id)?;
                let r = intersect_ranges(*r1, *r2)?;
                Some((*file_id, r))
            })
            .collect();

        return SearchScope::new(res);

        fn intersect_ranges(
            r1: Option<TextRange>,
            r2: Option<TextRange>,
        ) -> Option<Option<TextRange>> {
            match (r1, r2) {
                (None, r) | (r, None) => Some(r),
                (Some(r1), Some(r2)) => {
                    let r = r1.intersect(r2)?;
                    Some(Some(r))
                }
            }
        }
    }
}

impl IntoIterator for SearchScope {
    type Item = (FileId, Option<TextRange>);
    type IntoIter = std::collections::hash_map::IntoIter<FileId, Option<TextRange>>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl Definition {
    fn search_scope(&self, db: &RootDatabase) -> SearchScope {
        let _p = profile::span("search_scope");

        if let Definition::ModuleDef(hir::ModuleDef::BuiltinType(_)) = self {
            return SearchScope::crate_graph(db);
        }

        let module = match self.module(db) {
            Some(it) => it,
            None => return SearchScope::empty(),
        };
        let InFile { file_id, value: module_source } = module.definition_source(db);
        let file_id = file_id.original_file(db);

        if let Definition::Local(var) = self {
            let range = match var.parent(db) {
                DefWithBody::Function(f) => f.source(db).map(|src| src.value.syntax().text_range()),
                DefWithBody::Const(c) => c.source(db).map(|src| src.value.syntax().text_range()),
                DefWithBody::Static(s) => s.source(db).map(|src| src.value.syntax().text_range()),
            };
            return match range {
                Some(range) => SearchScope::file_range(FileRange { file_id, range }),
                None => SearchScope::single_file(file_id),
            };
        }

        if let Definition::GenericParam(hir::GenericParam::LifetimeParam(param)) = self {
            let range = match param.parent(db) {
                hir::GenericDef::Function(it) => {
                    it.source(db).map(|src| src.value.syntax().text_range())
                }
                hir::GenericDef::Adt(it) => match it {
                    hir::Adt::Struct(it) => {
                        it.source(db).map(|src| src.value.syntax().text_range())
                    }
                    hir::Adt::Union(it) => it.source(db).map(|src| src.value.syntax().text_range()),
                    hir::Adt::Enum(it) => it.source(db).map(|src| src.value.syntax().text_range()),
                },
                hir::GenericDef::Trait(it) => {
                    it.source(db).map(|src| src.value.syntax().text_range())
                }
                hir::GenericDef::TypeAlias(it) => {
                    it.source(db).map(|src| src.value.syntax().text_range())
                }
                hir::GenericDef::Impl(it) => {
                    it.source(db).map(|src| src.value.syntax().text_range())
                }
                hir::GenericDef::Variant(it) => {
                    it.source(db).map(|src| src.value.syntax().text_range())
                }
                hir::GenericDef::Const(it) => {
                    it.source(db).map(|src| src.value.syntax().text_range())
                }
            };
            return match range {
                Some(range) => SearchScope::file_range(FileRange { file_id, range }),
                None => SearchScope::single_file(file_id),
            };
        }

        if let Definition::Macro(macro_def) = self {
            if macro_def.kind() == hir::MacroKind::Declarative {
                return if macro_def.attrs(db).by_key("macro_export").exists() {
                    SearchScope::reverse_dependencies(db, module.krate())
                } else {
                    SearchScope::krate(db, module.krate())
                };
            }
        }

        let vis = self.visibility(db);
        if let Some(Visibility::Public) = vis {
            return SearchScope::reverse_dependencies(db, module.krate());
        }
        if let Some(Visibility::Module(module)) = vis {
            return SearchScope::module(db, module.into());
        }

        let range = match module_source {
            ModuleSource::Module(m) => Some(m.syntax().text_range()),
            ModuleSource::BlockExpr(b) => Some(b.syntax().text_range()),
            ModuleSource::SourceFile(_) => None,
        };
        match range {
            Some(range) => SearchScope::file_range(FileRange { file_id, range }),
            None => SearchScope::single_file(file_id),
        }
    }

    pub fn usages<'a>(&'a self, sema: &'a Semantics<RootDatabase>) -> FindUsages<'a> {
        FindUsages { def: self, sema, scope: None, include_self_kw_refs: false }
    }
}

pub struct FindUsages<'a> {
    def: &'a Definition,
    sema: &'a Semantics<'a, RootDatabase>,
    scope: Option<SearchScope>,
    include_self_kw_refs: bool,
}

impl<'a> FindUsages<'a> {
    pub fn include_self_kw_refs(mut self, include: bool) -> FindUsages<'a> {
        self.include_self_kw_refs = include;
        self
    }

    pub fn in_scope(self, scope: SearchScope) -> FindUsages<'a> {
        self.set_scope(Some(scope))
    }

    pub fn set_scope(mut self, scope: Option<SearchScope>) -> FindUsages<'a> {
        assert!(self.scope.is_none());
        self.scope = scope;
        self
    }

    pub fn at_least_one(self) -> bool {
        let mut found = false;
        self.search(&mut |_, _| {
            found = true;
            true
        });
        found
    }

    pub fn all(self) -> UsageSearchResult {
        let mut res = UsageSearchResult::default();
        self.search(&mut |file_id, reference| {
            res.references.entry(file_id).or_default().push(reference);
            false
        });
        res
    }

    fn search(self, sink: &mut dyn FnMut(FileId, FileReference) -> bool) {
        let _p = profile::span("FindUsages:search");
        let sema = self.sema;

        let search_scope = {
            let base = self.def.search_scope(sema.db);
            match &self.scope {
                None => base,
                Some(scope) => base.intersection(scope),
            }
        };

        let name = match self.def.name(sema.db) {
            Some(it) => it.to_string(),
            None => return,
        };

        let pat = name.as_str();
        let search_for_self = self.include_self_kw_refs;

        for (file_id, search_range) in search_scope {
            let text = sema.db.file_text(file_id);
            let search_range =
                search_range.unwrap_or_else(|| TextRange::up_to(TextSize::of(text.as_str())));

            let tree = Lazy::new(|| sema.parse(file_id).syntax().clone());

            let mut handle_match = |idx: usize| -> bool {
                let offset: TextSize = idx.try_into().unwrap();
                if !search_range.contains_inclusive(offset) {
                    return false;
                }

                if let Some(name) = sema.find_node_at_offset_with_descend(&tree, offset) {
                    match name {
                        ast::NameLike::NameRef(name_ref) => {
                            if self.found_name_ref(&name_ref, sink) {
                                return true;
                            }
                        }
                        ast::NameLike::Name(name) => {
                            if self.found_name(&name, sink) {
                                return true;
                            }
                        }
                        ast::NameLike::Lifetime(lifetime) => {
                            if self.found_lifetime(&lifetime, sink) {
                                return true;
                            }
                        }
                    }
                }

                return false;
            };

            for (idx, _) in text.match_indices(pat) {
                if handle_match(idx) {
                    return;
                }
            }

            if search_for_self {
                for (idx, _) in text.match_indices("Self") {
                    if handle_match(idx) {
                        return;
                    }
                }
            }
        }
    }

    fn found_lifetime(
        &self,
        lifetime: &ast::Lifetime,
        sink: &mut dyn FnMut(FileId, FileReference) -> bool,
    ) -> bool {
        match NameRefClass::classify_lifetime(self.sema, lifetime) {
            Some(NameRefClass::Definition(def)) if &def == self.def => {
                let FileRange { file_id, range } = self.sema.original_range(lifetime.syntax());
                let reference = FileReference {
                    range,
                    name: ast::NameLike::Lifetime(lifetime.clone()),
                    access: None,
                };
                sink(file_id, reference)
            }
            _ => false, // not a usage
        }
    }

    fn found_name_ref(
        &self,
        name_ref: &ast::NameRef,
        sink: &mut dyn FnMut(FileId, FileReference) -> bool,
    ) -> bool {
        match NameRefClass::classify(self.sema, &name_ref) {
            Some(NameRefClass::Definition(def)) if &def == self.def => {
                let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                let reference = FileReference {
                    range,
                    name: ast::NameLike::NameRef(name_ref.clone()),
                    access: reference_access(&def, &name_ref),
                };
                sink(file_id, reference)
            }
            Some(NameRefClass::Definition(Definition::SelfType(impl_))) => {
                let ty = impl_.self_ty(self.sema.db);

                if let Some(adt) = ty.as_adt() {
                    if &Definition::ModuleDef(ModuleDef::Adt(adt)) == self.def {
                        let FileRange { file_id, range } =
                            self.sema.original_range(name_ref.syntax());
                        let reference = FileReference {
                            range,
                            name: ast::NameLike::NameRef(name_ref.clone()),
                            access: None,
                        };
                        return sink(file_id, reference);
                    }
                }

                false
            }
            Some(NameRefClass::FieldShorthand { local_ref: local, field_ref: field }) => {
                let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                let reference = match self.def {
                    Definition::Field(_) if &field == self.def => FileReference {
                        range,
                        name: ast::NameLike::NameRef(name_ref.clone()),
                        access: reference_access(&field, &name_ref),
                    },
                    Definition::Local(l) if &local == l => FileReference {
                        range,
                        name: ast::NameLike::NameRef(name_ref.clone()),
                        access: reference_access(&Definition::Local(local), &name_ref),
                    },
                    _ => return false, // not a usage
                };
                sink(file_id, reference)
            }
            _ => false, // not a usage
        }
    }

    fn found_name(
        &self,
        name: &ast::Name,
        sink: &mut dyn FnMut(FileId, FileReference) -> bool,
    ) -> bool {
        match NameClass::classify(self.sema, name) {
            Some(NameClass::PatFieldShorthand { local_def: _, field_ref })
                if matches!(
                    self.def, Definition::Field(_) if &field_ref == self.def
                ) =>
            {
                let FileRange { file_id, range } = self.sema.original_range(name.syntax());
                let reference = FileReference {
                    range,
                    name: ast::NameLike::Name(name.clone()),
                    // FIXME: mutable patterns should have `Write` access
                    access: Some(ReferenceAccess::Read),
                };
                sink(file_id, reference)
            }
            Some(NameClass::ConstReference(def)) if *self.def == def => {
                let FileRange { file_id, range } = self.sema.original_range(name.syntax());
                let reference =
                    FileReference { range, name: ast::NameLike::Name(name.clone()), access: None };
                sink(file_id, reference)
            }
            _ => false, // not a usage
        }
    }
}

fn reference_access(def: &Definition, name_ref: &ast::NameRef) -> Option<ReferenceAccess> {
    // Only Locals and Fields have accesses for now.
    if !matches!(def, Definition::Local(_) | Definition::Field(_)) {
        return None;
    }

    let mode = name_ref.syntax().ancestors().find_map(|node| {
        match_ast! {
            match (node) {
                ast::BinExpr(expr) => {
                    if expr.op_kind()?.is_assignment() {
                        // If the variable or field ends on the LHS's end then it's a Write (covers fields and locals).
                        // FIXME: This is not terribly accurate.
                        if let Some(lhs) = expr.lhs() {
                            if lhs.syntax().text_range().end() == name_ref.syntax().text_range().end() {
                                return Some(ReferenceAccess::Write);
                            }
                        }
                    }
                    Some(ReferenceAccess::Read)
                },
                _ => None
            }
        }
    });

    // Default Locals and Fields to read
    mode.or(Some(ReferenceAccess::Read))
}
