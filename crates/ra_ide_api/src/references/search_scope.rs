//! Generally, `search_scope` returns files that might contain references for the element.
//! For `pub(crate)` things it's a crate, for `pub` things it's a crate and dependant crates.
//! In some cases, the location of the references is known to within a `TextRange`,
//! e.g. for things like local variables.
use std::mem;

use hir::{DefWithBody, HasSource, ModuleSource};
use ra_db::{FileId, SourceDatabase, SourceDatabaseExt};
use ra_prof::profile;
use ra_syntax::{AstNode, TextRange};
use rustc_hash::FxHashMap;

use crate::db::RootDatabase;

use super::{NameDefinition, NameKind};

pub struct SearchScope {
    entries: FxHashMap<FileId, Option<TextRange>>,
}

impl SearchScope {
    fn new(entries: FxHashMap<FileId, Option<TextRange>>) -> SearchScope {
        SearchScope { entries }
    }
    pub fn single_file(file: FileId) -> SearchScope {
        SearchScope::new(std::iter::once((file, None)).collect())
    }
    pub(crate) fn intersection(&self, other: &SearchScope) -> SearchScope {
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

impl NameDefinition {
    pub(crate) fn search_scope(&self, db: &RootDatabase) -> SearchScope {
        let _p = profile("search_scope");

        let module_src = self.container.definition_source(db);
        let file_id = module_src.file_id.original_file(db);

        if let NameKind::Pat((def, _)) = self.kind {
            let mut res = FxHashMap::default();
            let range = match def {
                DefWithBody::Function(f) => f.source(db).ast.syntax().text_range(),
                DefWithBody::Const(c) => c.source(db).ast.syntax().text_range(),
                DefWithBody::Static(s) => s.source(db).ast.syntax().text_range(),
            };
            res.insert(file_id, Some(range));
            return SearchScope::new(res);
        }

        let vis =
            self.visibility.as_ref().map(|v| v.syntax().to_string()).unwrap_or("".to_string());

        if vis.as_str() == "pub(super)" {
            if let Some(parent_module) = self.container.parent(db) {
                let mut res = FxHashMap::default();
                let parent_src = parent_module.definition_source(db);
                let file_id = parent_src.file_id.original_file(db);

                match parent_src.ast {
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
                let krate = self.container.krate();
                let crate_graph = db.crate_graph();
                for crate_id in crate_graph.iter() {
                    let mut crate_deps = crate_graph.dependencies(crate_id);
                    if crate_deps.any(|dep| dep.crate_id() == krate.crate_id()) {
                        let root_file = crate_graph.crate_root(crate_id);
                        let source_root_id = db.file_source_root(root_file);
                        let source_root = db.source_root(source_root_id);
                        res.extend(source_root.walk().map(|id| (id, None)));
                    }
                }
                return SearchScope::new(res);
            }
        }

        let mut res = FxHashMap::default();
        let range = match module_src.ast {
            ModuleSource::Module(m) => Some(m.syntax().text_range()),
            ModuleSource::SourceFile(_) => None,
        };
        res.insert(file_id, range);
        SearchScope::new(res)
    }
}
