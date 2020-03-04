//! Generally, `search_scope` returns files that might contain references for the element.
//! For `pub(crate)` things it's a crate, for `pub` things it's a crate and dependant crates.
//! In some cases, the location of the references is known to within a `TextRange`,
//! e.g. for things like local variables.
use std::mem;

use hir::{DefWithBody, HasSource, ModuleSource, Semantics};
use once_cell::unsync::Lazy;
use ra_db::{FileId, FileRange, SourceDatabaseExt};
use ra_prof::profile;
use ra_syntax::{
    algo::find_node_at_offset, ast, match_ast, AstNode, TextRange, TextUnit, TokenAtOffset,
};
use rustc_hash::FxHashMap;

use crate::{
    defs::{classify_name_ref, Definition},
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
    StructLiteral,
    Other,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ReferenceAccess {
    Read,
    Write,
}

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

    pub fn for_def(def: &Definition, db: &RootDatabase) -> SearchScope {
        let _p = profile("search_scope");
        let module = match def.module(db) {
            Some(it) => it,
            None => return SearchScope::empty(),
        };
        let module_src = module.definition_source(db);
        let file_id = module_src.file_id.original_file(db);

        if let Definition::Local(var) = def {
            let range = match var.parent(db) {
                DefWithBody::Function(f) => f.source(db).value.syntax().text_range(),
                DefWithBody::Const(c) => c.source(db).value.syntax().text_range(),
                DefWithBody::Static(s) => s.source(db).value.syntax().text_range(),
            };
            let mut res = FxHashMap::default();
            res.insert(file_id, Some(range));
            return SearchScope::new(res);
        }

        let vis = def.visibility(db).as_ref().map(|v| v.syntax().to_string()).unwrap_or_default();

        if vis.as_str() == "pub(super)" {
            if let Some(parent_module) = module.parent(db) {
                let mut res = FxHashMap::default();
                let parent_src = parent_module.definition_source(db);
                let file_id = parent_src.file_id.original_file(db);

                match parent_src.value {
                    ModuleSource::Module(m) => {
                        let range = Some(m.syntax().text_range());
                        res.insert(file_id, range);
                    }
                    ModuleSource::SourceFile(_) => {
                        res.insert(file_id, None);
                        res.extend(parent_module.children(db).map(|m| {
                            let src = m.definition_source(db);
                            (src.file_id.original_file(db), None)
                        }));
                    }
                }
                return SearchScope::new(res);
            }
        }

        if vis.as_str() != "" {
            let source_root_id = db.file_source_root(file_id);
            let source_root = db.source_root(source_root_id);
            let mut res = source_root.walk().map(|id| (id, None)).collect::<FxHashMap<_, _>>();

            // FIXME: add "pub(in path)"

            if vis.as_str() == "pub(crate)" {
                return SearchScope::new(res);
            }
            if vis.as_str() == "pub" {
                let krate = module.krate();
                for rev_dep in krate.reverse_dependencies(db) {
                    let root_file = rev_dep.root_file(db);
                    let source_root_id = db.file_source_root(root_file);
                    let source_root = db.source_root(source_root_id);
                    res.extend(source_root.walk().map(|id| (id, None)));
                }
                return SearchScope::new(res);
            }
        }

        let mut res = FxHashMap::default();
        let range = match module_src.value {
            ModuleSource::Module(m) => Some(m.syntax().text_range()),
            ModuleSource::SourceFile(_) => None,
        };
        res.insert(file_id, range);
        SearchScope::new(res)
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
                    let r = r1.intersection(&r2)?;
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
    pub fn find_usages(
        &self,
        db: &RootDatabase,
        search_scope: Option<SearchScope>,
    ) -> Vec<Reference> {
        let _p = profile("Definition::find_usages");

        let search_scope = {
            let base = SearchScope::for_def(self, db);
            match search_scope {
                None => base,
                Some(scope) => base.intersection(&scope),
            }
        };

        let name = match self.name(db) {
            None => return Vec::new(),
            Some(it) => it.to_string(),
        };

        let pat = name.as_str();
        let mut refs = vec![];

        for (file_id, search_range) in search_scope {
            let text = db.file_text(file_id);
            let search_range =
                search_range.unwrap_or(TextRange::offset_len(0.into(), TextUnit::of_str(&text)));

            let sema = Semantics::new(db);
            let tree = Lazy::new(|| sema.parse(file_id).syntax().clone());

            for (idx, _) in text.match_indices(pat) {
                let offset = TextUnit::from_usize(idx);
                if !search_range.contains_inclusive(offset) {
                    // tested_by!(search_filters_by_range);
                    continue;
                }

                let name_ref =
                    if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(&tree, offset) {
                        name_ref
                    } else {
                        // Handle macro token cases
                        let token = match tree.token_at_offset(offset) {
                            TokenAtOffset::None => continue,
                            TokenAtOffset::Single(t) => t,
                            TokenAtOffset::Between(_, t) => t,
                        };
                        let expanded = sema.descend_into_macros(token);
                        match ast::NameRef::cast(expanded.parent()) {
                            Some(name_ref) => name_ref,
                            _ => continue,
                        }
                    };

                // FIXME: reuse sb
                // See https://github.com/rust-lang/rust/pull/68198#issuecomment-574269098

                if let Some(d) = classify_name_ref(&sema, &name_ref) {
                    let d = d.definition();
                    if &d == self {
                        let kind = if is_record_lit_name_ref(&name_ref)
                            || is_call_expr_name_ref(&name_ref)
                        {
                            ReferenceKind::StructLiteral
                        } else {
                            ReferenceKind::Other
                        };

                        let file_range = sema.original_range(name_ref.syntax());
                        refs.push(Reference {
                            file_range,
                            kind,
                            access: reference_access(&d, &name_ref),
                        });
                    }
                }
            }
        }
        refs
    }
}

fn reference_access(def: &Definition, name_ref: &ast::NameRef) -> Option<ReferenceAccess> {
    // Only Locals and Fields have accesses for now.
    match def {
        Definition::Local(_) | Definition::StructField(_) => {}
        _ => return None,
    };

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
                _ => {None}
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
        .find_map(ast::RecordLit::cast)
        .and_then(|l| l.path())
        .and_then(|p| p.segment())
        .map(|p| p.name_ref().as_ref() == Some(name_ref))
        .unwrap_or(false)
}
