//! Implementation of find-usages functionality.
//!
//! It is based on the standard ide trick: first, we run a fast text search to
//! get a super-set of matches. Then, we we confirm each match using precise
//! name resolution.

use std::{convert::TryInto, mem};

use base_db::{FileId, FileRange, SourceDatabaseExt};
use hir::{DefWithBody, HasSource, Module, ModuleSource, Semantics, Visibility};
use once_cell::unsync::Lazy;
use rustc_hash::FxHashMap;
use syntax::{ast, match_ast, AstNode, TextRange, TextSize};

use crate::{
    defs::{classify_name_ref, Definition, NameRefClass},
    RootDatabase,
};

#[derive(Debug, Clone)]
pub struct Reference {
    pub file_range: FileRange,
    pub kind: ReferenceKind,
    pub access: Option<ReferenceAccess>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReferenceKind {
    FieldShorthandForField,
    FieldShorthandForLocal,
    StructLiteral,
    Other,
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

    pub fn empty() -> SearchScope {
        SearchScope::new(FxHashMap::default())
    }

    pub fn single_file(file: FileId) -> SearchScope {
        SearchScope::new(std::iter::once((file, None)).collect())
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
        let module = match self.module(db) {
            Some(it) => it,
            None => return SearchScope::empty(),
        };
        let module_src = module.definition_source(db);
        let file_id = module_src.file_id.original_file(db);

        if let Definition::Local(var) = self {
            let range = match var.parent(db) {
                DefWithBody::Function(f) => f.source(db).value.syntax().text_range(),
                DefWithBody::Const(c) => c.source(db).value.syntax().text_range(),
                DefWithBody::Static(s) => s.source(db).value.syntax().text_range(),
            };
            let mut res = FxHashMap::default();
            res.insert(file_id, Some(range));
            return SearchScope::new(res);
        }

        let vis = self.visibility(db);

        if let Some(Visibility::Module(module)) = vis.and_then(|it| it.into()) {
            let module: Module = module.into();
            let mut res = FxHashMap::default();

            let mut to_visit = vec![module];
            let mut is_first = true;
            while let Some(module) = to_visit.pop() {
                let src = module.definition_source(db);
                let file_id = src.file_id.original_file(db);
                match src.value {
                    ModuleSource::Module(m) => {
                        if is_first {
                            let range = Some(m.syntax().text_range());
                            res.insert(file_id, range);
                        } else {
                            // We have already added the enclosing file to the search scope,
                            // so do nothing.
                        }
                    }
                    ModuleSource::SourceFile(_) => {
                        res.insert(file_id, None);
                    }
                };
                is_first = false;
                to_visit.extend(module.children(db));
            }

            return SearchScope::new(res);
        }

        if let Some(Visibility::Public) = vis {
            let source_root_id = db.file_source_root(file_id);
            let source_root = db.source_root(source_root_id);
            let mut res = source_root.iter().map(|id| (id, None)).collect::<FxHashMap<_, _>>();

            let krate = module.krate();
            for rev_dep in krate.reverse_dependencies(db) {
                let root_file = rev_dep.root_file(db);
                let source_root_id = db.file_source_root(root_file);
                let source_root = db.source_root(source_root_id);
                res.extend(source_root.iter().map(|id| (id, None)));
            }
            return SearchScope::new(res);
        }

        let mut res = FxHashMap::default();
        let range = match module_src.value {
            ModuleSource::Module(m) => Some(m.syntax().text_range()),
            ModuleSource::SourceFile(_) => None,
        };
        res.insert(file_id, range);
        SearchScope::new(res)
    }

    pub fn usages<'a>(&'a self, sema: &'a Semantics<RootDatabase>) -> FindUsages<'a> {
        FindUsages { def: self, sema, scope: None }
    }
}

pub struct FindUsages<'a> {
    def: &'a Definition,
    sema: &'a Semantics<'a, RootDatabase>,
    scope: Option<SearchScope>,
}

impl<'a> FindUsages<'a> {
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
        self.search(&mut |_reference| {
            found = true;
            true
        });
        found
    }

    pub fn all(self) -> Vec<Reference> {
        let mut res = Vec::new();
        self.search(&mut |reference| {
            res.push(reference);
            false
        });
        res
    }

    fn search(self, sink: &mut dyn FnMut(Reference) -> bool) {
        let _p = profile::span("FindUsages:search");
        let sema = self.sema;

        let search_scope = {
            let base = self.def.search_scope(sema.db);
            match self.scope {
                None => base,
                Some(scope) => base.intersection(&scope),
            }
        };

        let name = match self.def.name(sema.db) {
            Some(it) => it.to_string(),
            None => return,
        };

        let pat = name.as_str();
        for (file_id, search_range) in search_scope {
            let text = sema.db.file_text(file_id);
            let search_range =
                search_range.unwrap_or(TextRange::up_to(TextSize::of(text.as_str())));

            let tree = Lazy::new(|| sema.parse(file_id).syntax().clone());

            for (idx, _) in text.match_indices(pat) {
                let offset: TextSize = idx.try_into().unwrap();
                if !search_range.contains_inclusive(offset) {
                    continue;
                }

                let name_ref: ast::NameRef =
                    match sema.find_node_at_offset_with_descend(&tree, offset) {
                        Some(it) => it,
                        None => continue,
                    };

                match classify_name_ref(&sema, &name_ref) {
                    Some(NameRefClass::Definition(def)) if &def == self.def => {
                        let kind = if is_record_lit_name_ref(&name_ref)
                            || is_call_expr_name_ref(&name_ref)
                        {
                            ReferenceKind::StructLiteral
                        } else {
                            ReferenceKind::Other
                        };

                        let reference = Reference {
                            file_range: sema.original_range(name_ref.syntax()),
                            kind,
                            access: reference_access(&def, &name_ref),
                        };
                        if sink(reference) {
                            return;
                        }
                    }
                    Some(NameRefClass::FieldShorthand { local, field }) => {
                        let reference = match self.def {
                            Definition::Field(_) if &field == self.def => Reference {
                                file_range: self.sema.original_range(name_ref.syntax()),
                                kind: ReferenceKind::FieldShorthandForField,
                                access: reference_access(&field, &name_ref),
                            },
                            Definition::Local(l) if &local == l => Reference {
                                file_range: self.sema.original_range(name_ref.syntax()),
                                kind: ReferenceKind::FieldShorthandForLocal,
                                access: reference_access(&Definition::Local(local), &name_ref),
                            },
                            _ => continue, // not a usage
                        };
                        if sink(reference) {
                            return;
                        }
                    }
                    _ => {} // not a usage
                }
            }
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

fn is_call_expr_name_ref(name_ref: &ast::NameRef) -> bool {
    name_ref
        .syntax()
        .ancestors()
        .find_map(ast::CallExpr::cast)
        .and_then(|c| match c.expr()? {
            ast::Expr::PathExpr(p) => {
                Some(p.path()?.segment()?.name_ref().as_ref() == Some(name_ref))
            }
            _ => None,
        })
        .unwrap_or(false)
}

fn is_record_lit_name_ref(name_ref: &ast::NameRef) -> bool {
    name_ref
        .syntax()
        .ancestors()
        .find_map(ast::RecordExpr::cast)
        .and_then(|l| l.path())
        .and_then(|p| p.segment())
        .map(|p| p.name_ref().as_ref() == Some(name_ref))
        .unwrap_or(false)
}
