//! Implementation of find-usages functionality.
//!
//! It is based on the standard ide trick: first, we run a fast text search to
//! get a super-set of matches. Then, we we confirm each match using precise
//! name resolution.

use std::{convert::TryInto, mem, sync::Arc};

use base_db::{FileId, FileRange, SourceDatabase, SourceDatabaseExt};
use hir::{
    AsAssocItem, DefWithBody, HasAttrs, HasSource, InFile, ModuleSource, Semantics, Visibility,
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

    pub fn iter(&self) -> impl Iterator<Item = (&FileId, &[FileReference])> + '_ {
        self.references.iter().map(|(file_id, refs)| (file_id, &**refs))
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
    pub category: Option<ReferenceCategory>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ReferenceCategory {
    // FIXME: Add this variant and delete the `retain_adt_literal_usages` function.
    // Create
    Write,
    Read,
    // FIXME: Some day should be able to search in doc comments. Would probably
    // need to switch from enum to bitflags then?
    // DocComment
}

/// Generally, `search_scope` returns files that might contain references for the element.
/// For `pub(crate)` things it's a crate, for `pub` things it's a crate and dependant crates.
/// In some cases, the location of the references is known to within a `TextRange`,
/// e.g. for things like local variables.
#[derive(Clone, Debug)]
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

        if let Definition::BuiltinType(_) = self {
            return SearchScope::crate_graph(db);
        }

        // def is crate root
        // FIXME: We don't do searches for crates currently, as a crate does not actually have a single name
        if let &Definition::Module(module) = self {
            if module.is_crate_root(db) {
                return SearchScope::reverse_dependencies(db, module.krate());
            }
        }

        let module = match self.module(db) {
            Some(it) => it,
            None => return SearchScope::empty(),
        };
        let InFile { file_id, value: module_source } = module.definition_source(db);
        let file_id = file_id.original_file(db);

        if let Definition::Local(var) = self {
            let def = match var.parent(db) {
                DefWithBody::Function(f) => f.source(db).map(|src| src.syntax().cloned()),
                DefWithBody::Const(c) => c.source(db).map(|src| src.syntax().cloned()),
                DefWithBody::Static(s) => s.source(db).map(|src| src.syntax().cloned()),
            };
            return match def {
                Some(def) => SearchScope::file_range(def.as_ref().original_file_range(db)),
                None => SearchScope::single_file(file_id),
            };
        }

        if let Definition::SelfType(impl_) = self {
            return match impl_.source(db).map(|src| src.syntax().cloned()) {
                Some(def) => SearchScope::file_range(def.as_ref().original_file_range(db)),
                None => SearchScope::single_file(file_id),
            };
        }

        if let Definition::GenericParam(hir::GenericParam::LifetimeParam(param)) = self {
            let def = match param.parent(db) {
                hir::GenericDef::Function(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::Adt(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::Trait(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::TypeAlias(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::Impl(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::Variant(it) => it.source(db).map(|src| src.syntax().cloned()),
                hir::GenericDef::Const(it) => it.source(db).map(|src| src.syntax().cloned()),
            };
            return match def {
                Some(def) => SearchScope::file_range(def.as_ref().original_file_range(db)),
                None => SearchScope::single_file(file_id),
            };
        }

        if let Definition::Macro(macro_def) = self {
            return match macro_def.kind(db) {
                hir::MacroKind::Declarative => {
                    if macro_def.attrs(db).by_key("macro_export").exists() {
                        SearchScope::reverse_dependencies(db, module.krate())
                    } else {
                        SearchScope::krate(db, module.krate())
                    }
                }
                hir::MacroKind::BuiltIn => SearchScope::crate_graph(db),
                // FIXME: We don't actually see derives in derive attributes as these do not
                // expand to something that references the derive macro in the output.
                // We could get around this by emitting dummy `use DeriveMacroPathHere as _;` items maybe?
                hir::MacroKind::Derive | hir::MacroKind::Attr | hir::MacroKind::ProcMacro => {
                    SearchScope::reverse_dependencies(db, module.krate())
                }
            };
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

    pub fn usages<'a>(self, sema: &'a Semantics<RootDatabase>) -> FindUsages<'a> {
        FindUsages {
            local_repr: match self {
                Definition::Local(local) => Some(local.representative(sema.db)),
                _ => None,
            },
            def: self,
            sema,
            scope: None,
            include_self_kw_refs: None,
            search_self_mod: false,
        }
    }
}

#[derive(Clone)]
pub struct FindUsages<'a> {
    def: Definition,
    sema: &'a Semantics<'a, RootDatabase>,
    scope: Option<SearchScope>,
    include_self_kw_refs: Option<hir::Type>,
    local_repr: Option<hir::Local>,
    search_self_mod: bool,
}

impl<'a> FindUsages<'a> {
    /// Enable searching for `Self` when the definition is a type or `self` for modules.
    pub fn include_self_refs(mut self) -> FindUsages<'a> {
        self.include_self_kw_refs = def_to_ty(self.sema, &self.def);
        self.search_self_mod = true;
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

    pub fn at_least_one(&self) -> bool {
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

    fn search(&self, sink: &mut dyn FnMut(FileId, FileReference) -> bool) {
        let _p = profile::span("FindUsages:search");
        let sema = self.sema;

        let search_scope = {
            let base = self.def.search_scope(sema.db);
            match &self.scope {
                None => base,
                Some(scope) => base.intersection(scope),
            }
        };

        let name = match self.def {
            // special case crate modules as these do not have a proper name
            Definition::Module(module) if module.is_crate_root(self.sema.db) => {
                // FIXME: This assumes the crate name is always equal to its display name when it really isn't
                module
                    .krate()
                    .display_name(self.sema.db)
                    .map(|crate_name| crate_name.crate_name().as_smol_str().clone())
            }
            _ => {
                let self_kw_refs = || {
                    self.include_self_kw_refs.as_ref().and_then(|ty| {
                        ty.as_adt()
                            .map(|adt| adt.name(self.sema.db))
                            .or_else(|| ty.as_builtin().map(|builtin| builtin.name()))
                    })
                };
                self.def.name(sema.db).or_else(self_kw_refs).map(|it| it.to_smol_str())
            }
        };
        let name = match &name {
            Some(s) => s.as_str(),
            None => return,
        };

        // these can't be closures because rust infers the lifetimes wrong ...
        fn match_indices<'a>(
            text: &'a str,
            name: &'a str,
            search_range: TextRange,
        ) -> impl Iterator<Item = TextSize> + 'a {
            text.match_indices(name).filter_map(move |(idx, _)| {
                let offset: TextSize = idx.try_into().unwrap();
                if !search_range.contains_inclusive(offset) {
                    return None;
                }
                Some(offset)
            })
        }
        fn scope_files<'a>(
            sema: &'a Semantics<RootDatabase>,
            scope: &'a SearchScope,
        ) -> impl Iterator<Item = (Arc<String>, FileId, TextRange)> + 'a {
            scope.entries.iter().map(|(&file_id, &search_range)| {
                let text = sema.db.file_text(file_id);
                let search_range =
                    search_range.unwrap_or_else(|| TextRange::up_to(TextSize::of(text.as_str())));

                (text, file_id, search_range)
            })
        }

        for (text, file_id, search_range) in scope_files(sema, &search_scope) {
            let tree = Lazy::new(move || sema.parse(file_id).syntax().clone());

            // Search for occurrences of the items name
            for offset in match_indices(&text, name, search_range) {
                for name in sema.find_nodes_at_offset_with_descend(&tree, offset) {
                    if match name {
                        ast::NameLike::NameRef(name_ref) => self.found_name_ref(&name_ref, sink),
                        ast::NameLike::Name(name) => self.found_name(&name, sink),
                        ast::NameLike::Lifetime(lifetime) => self.found_lifetime(&lifetime, sink),
                    } {
                        return;
                    }
                }
            }
            // Search for occurrences of the `Self` referring to our type
            if let Some(self_ty) = &self.include_self_kw_refs {
                for offset in match_indices(&text, "Self", search_range) {
                    for name_ref in sema.find_nodes_at_offset_with_descend(&tree, offset) {
                        if self.found_self_ty_name_ref(self_ty, &name_ref, sink) {
                            return;
                        }
                    }
                }
            }
        }

        // Search for `super` and `crate` resolving to our module
        match self.def {
            Definition::Module(module) => {
                let scope = search_scope.intersection(&SearchScope::module(self.sema.db, module));

                let is_crate_root = module.is_crate_root(self.sema.db);

                for (text, file_id, search_range) in scope_files(sema, &scope) {
                    let tree = Lazy::new(move || sema.parse(file_id).syntax().clone());

                    for offset in match_indices(&text, "super", search_range) {
                        for name_ref in sema.find_nodes_at_offset_with_descend(&tree, offset) {
                            if self.found_name_ref(&name_ref, sink) {
                                return;
                            }
                        }
                    }
                    if is_crate_root {
                        for offset in match_indices(&text, "crate", search_range) {
                            for name_ref in sema.find_nodes_at_offset_with_descend(&tree, offset) {
                                if self.found_name_ref(&name_ref, sink) {
                                    return;
                                }
                            }
                        }
                    }
                }
            }
            _ => (),
        }

        // search for module `self` references in our module's definition source
        match self.def {
            Definition::Module(module) if self.search_self_mod => {
                let src = module.definition_source(sema.db);
                let file_id = src.file_id.original_file(sema.db);
                let (file_id, search_range) = match src.value {
                    ModuleSource::Module(m) => (file_id, Some(m.syntax().text_range())),
                    ModuleSource::BlockExpr(b) => (file_id, Some(b.syntax().text_range())),
                    ModuleSource::SourceFile(_) => (file_id, None),
                };

                let search_range = if let Some(&range) = search_scope.entries.get(&file_id) {
                    match (range, search_range) {
                        (None, range) | (range, None) => range,
                        (Some(range), Some(search_range)) => match range.intersect(search_range) {
                            Some(range) => Some(range),
                            None => return,
                        },
                    }
                } else {
                    return;
                };

                let text = sema.db.file_text(file_id);
                let search_range =
                    search_range.unwrap_or_else(|| TextRange::up_to(TextSize::of(text.as_str())));

                let tree = Lazy::new(|| sema.parse(file_id).syntax().clone());

                for offset in match_indices(&text, "self", search_range) {
                    for name_ref in sema.find_nodes_at_offset_with_descend(&tree, offset) {
                        if self.found_self_module_name_ref(&name_ref, sink) {
                            return;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn found_self_ty_name_ref(
        &self,
        self_ty: &hir::Type,
        name_ref: &ast::NameRef,
        sink: &mut dyn FnMut(FileId, FileReference) -> bool,
    ) -> bool {
        match NameRefClass::classify(self.sema, name_ref) {
            Some(NameRefClass::Definition(Definition::SelfType(impl_)))
                if impl_.self_ty(self.sema.db) == *self_ty =>
            {
                let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                let reference = FileReference {
                    range,
                    name: ast::NameLike::NameRef(name_ref.clone()),
                    category: None,
                };
                sink(file_id, reference)
            }
            _ => false,
        }
    }

    fn found_self_module_name_ref(
        &self,
        name_ref: &ast::NameRef,
        sink: &mut dyn FnMut(FileId, FileReference) -> bool,
    ) -> bool {
        match NameRefClass::classify(self.sema, name_ref) {
            Some(NameRefClass::Definition(def @ Definition::Module(_))) if def == self.def => {
                let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                let reference = FileReference {
                    range,
                    name: ast::NameLike::NameRef(name_ref.clone()),
                    category: None,
                };
                sink(file_id, reference)
            }
            _ => false,
        }
    }

    fn found_lifetime(
        &self,
        lifetime: &ast::Lifetime,
        sink: &mut dyn FnMut(FileId, FileReference) -> bool,
    ) -> bool {
        match NameRefClass::classify_lifetime(self.sema, lifetime) {
            Some(NameRefClass::Definition(def)) if def == self.def => {
                let FileRange { file_id, range } = self.sema.original_range(lifetime.syntax());
                let reference = FileReference {
                    range,
                    name: ast::NameLike::Lifetime(lifetime.clone()),
                    category: None,
                };
                sink(file_id, reference)
            }
            _ => false,
        }
    }

    fn found_name_ref(
        &self,
        name_ref: &ast::NameRef,
        sink: &mut dyn FnMut(FileId, FileReference) -> bool,
    ) -> bool {
        match NameRefClass::classify(self.sema, name_ref) {
            Some(NameRefClass::Definition(def @ Definition::Local(local)))
                if matches!(
                    self.local_repr, Some(repr) if repr == local.representative(self.sema.db)
                ) =>
            {
                let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                let reference = FileReference {
                    range,
                    name: ast::NameLike::NameRef(name_ref.clone()),
                    category: ReferenceCategory::new(&def, name_ref),
                };
                sink(file_id, reference)
            }
            Some(NameRefClass::Definition(def)) if def == self.def => {
                let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                let reference = FileReference {
                    range,
                    name: ast::NameLike::NameRef(name_ref.clone()),
                    category: ReferenceCategory::new(&def, name_ref),
                };
                sink(file_id, reference)
            }
            Some(NameRefClass::Definition(def)) if self.include_self_kw_refs.is_some() => {
                if self.include_self_kw_refs == def_to_ty(self.sema, &def) {
                    let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                    let reference = FileReference {
                        range,
                        name: ast::NameLike::NameRef(name_ref.clone()),
                        category: ReferenceCategory::new(&def, name_ref),
                    };
                    sink(file_id, reference)
                } else {
                    false
                }
            }
            Some(NameRefClass::FieldShorthand { local_ref: local, field_ref: field }) => {
                let field = Definition::Field(field);
                let FileRange { file_id, range } = self.sema.original_range(name_ref.syntax());
                let access = match self.def {
                    Definition::Field(_) if field == self.def => {
                        ReferenceCategory::new(&field, name_ref)
                    }
                    Definition::Local(_) if matches!(self.local_repr, Some(repr) if repr == local.representative(self.sema.db)) => {
                        ReferenceCategory::new(&Definition::Local(local), name_ref)
                    }
                    _ => return false,
                };
                let reference = FileReference {
                    range,
                    name: ast::NameLike::NameRef(name_ref.clone()),
                    category: access,
                };
                sink(file_id, reference)
            }
            _ => false,
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
                    self.def, Definition::Field(_) if Definition::Field(field_ref) == self.def
                ) =>
            {
                let FileRange { file_id, range } = self.sema.original_range(name.syntax());
                let reference = FileReference {
                    range,
                    name: ast::NameLike::Name(name.clone()),
                    // FIXME: mutable patterns should have `Write` access
                    category: Some(ReferenceCategory::Read),
                };
                sink(file_id, reference)
            }
            Some(NameClass::ConstReference(def)) if self.def == def => {
                let FileRange { file_id, range } = self.sema.original_range(name.syntax());
                let reference = FileReference {
                    range,
                    name: ast::NameLike::Name(name.clone()),
                    category: None,
                };
                sink(file_id, reference)
            }
            Some(NameClass::Definition(def @ Definition::Local(local))) if def != self.def => {
                if matches!(
                    self.local_repr,
                    Some(repr) if local.representative(self.sema.db) == repr
                ) {
                    let FileRange { file_id, range } = self.sema.original_range(name.syntax());
                    let reference = FileReference {
                        range,
                        name: ast::NameLike::Name(name.clone()),
                        category: None,
                    };
                    return sink(file_id, reference);
                }
                false
            }
            // Resolve trait impl function definitions to the trait definition's version if self.def is the trait definition's
            Some(NameClass::Definition(def)) if def != self.def => {
                /* poor man's try block */
                (|| {
                    let this_trait = self
                        .def
                        .as_assoc_item(self.sema.db)?
                        .containing_trait_or_trait_impl(self.sema.db)?;
                    let trait_ = def
                        .as_assoc_item(self.sema.db)?
                        .containing_trait_or_trait_impl(self.sema.db)?;
                    (trait_ == this_trait && self.def.name(self.sema.db) == def.name(self.sema.db))
                        .then(|| {
                            let FileRange { file_id, range } =
                                self.sema.original_range(name.syntax());
                            let reference = FileReference {
                                range,
                                name: ast::NameLike::Name(name.clone()),
                                category: None,
                            };
                            sink(file_id, reference)
                        })
                })()
                .unwrap_or(false)
            }
            _ => false,
        }
    }
}

fn def_to_ty(sema: &Semantics<RootDatabase>, def: &Definition) -> Option<hir::Type> {
    match def {
        Definition::Adt(adt) => Some(adt.ty(sema.db)),
        Definition::TypeAlias(it) => Some(it.ty(sema.db)),
        Definition::BuiltinType(it) => {
            let graph = sema.db.crate_graph();
            let krate = graph.iter().next()?;
            let root_file = graph[krate].root_file_id;
            let module = sema.to_module_def(root_file)?;
            Some(it.ty(sema.db, module))
        }
        Definition::SelfType(it) => Some(it.self_ty(sema.db)),
        _ => None,
    }
}

impl ReferenceCategory {
    fn new(def: &Definition, r: &ast::NameRef) -> Option<ReferenceCategory> {
        // Only Locals and Fields have accesses for now.
        if !matches!(def, Definition::Local(_) | Definition::Field(_)) {
            return None;
        }

        let mode = r.syntax().ancestors().find_map(|node| {
        match_ast! {
            match node {
                ast::BinExpr(expr) => {
                    if matches!(expr.op_kind()?, ast::BinaryOp::Assignment { .. }) {
                        // If the variable or field ends on the LHS's end then it's a Write (covers fields and locals).
                        // FIXME: This is not terribly accurate.
                        if let Some(lhs) = expr.lhs() {
                            if lhs.syntax().text_range().end() == r.syntax().text_range().end() {
                                return Some(ReferenceCategory::Write);
                            }
                        }
                    }
                    Some(ReferenceCategory::Read)
                },
                _ => None
            }
        }
    });

        // Default Locals and Fields to read
        mode.or(Some(ReferenceCategory::Read))
    }
}
