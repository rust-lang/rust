//! FIXME: write short doc here

use hir::{DefWithBody, HasSource, ModuleSource};
use ra_db::{FileId, SourceDatabase};
use ra_syntax::{AstNode, TextRange};

use crate::db::RootDatabase;

use super::{NameDefinition, NameKind};

pub(crate) struct SearchScope {
    pub files: Vec<(FileId, Option<TextRange>)>,
}

impl NameDefinition {
    pub(crate) fn scope(&self, db: &RootDatabase) -> SearchScope {
        let module_src = self.container.definition_source(db);
        let file_id = module_src.file_id.original_file(db);

        if let NameKind::Pat((def, _)) = self.kind {
            let range = match def {
                DefWithBody::Function(f) => f.source(db).ast.syntax().text_range(),
                DefWithBody::Const(c) => c.source(db).ast.syntax().text_range(),
                DefWithBody::Static(s) => s.source(db).ast.syntax().text_range(),
            };
            return SearchScope { files: vec![(file_id, Some(range))] };
        }

        if let Some(ref vis) = self.visibility {
            let source_root_id = db.file_source_root(file_id);
            let source_root = db.source_root(source_root_id);
            let mut files = source_root.walk().map(|id| (id.into(), None)).collect::<Vec<_>>();

            if vis.syntax().to_string().as_str() == "pub(crate)" {
                return SearchScope { files };
            }
            if vis.syntax().to_string().as_str() == "pub" {
                let krate = self.container.krate(db).unwrap();
                let crate_graph = db.crate_graph();

                for crate_id in crate_graph.iter() {
                    let mut crate_deps = crate_graph.dependencies(crate_id);

                    if crate_deps.any(|dep| dep.crate_id() == krate.crate_id()) {
                        let root_file = crate_graph.crate_root(crate_id);
                        let source_root_id = db.file_source_root(root_file);
                        let source_root = db.source_root(source_root_id);
                        files.extend(source_root.walk().map(|id| (id.into(), None)));
                    }
                }

                return SearchScope { files };
            }
            // FIXME: "pub(super)", "pub(in path)"
        }

        let range = match module_src.ast {
            ModuleSource::Module(m) => Some(m.syntax().text_range()),
            ModuleSource::SourceFile(_) => None,
        };
        SearchScope { files: vec![(file_id, range)] }
    }
}
