use hir::{
    source_binder::ReferenceDescriptor, DefWithBody, HasSource, ModuleSource, SourceAnalyzer,
};
use ra_db::{FileId, SourceDatabase};
use ra_syntax::{algo::find_node_at_offset, ast, AstNode, SourceFile, TextRange, TextUnit};

use crate::{
    db::RootDatabase,
    name_kind::{classify_name_ref, Definition, NameKind},
};

pub(crate) struct SearchScope {
    pub scope: Vec<(FileId, Option<TextRange>)>,
}

pub(crate) fn find_refs(
    db: &RootDatabase,
    def: Definition,
    name: String,
) -> Vec<ReferenceDescriptor> {
    let pat = name.as_str();
    let scope = def.scope(db).scope;
    let mut refs = vec![];

    let is_match = |file_id: FileId, name_ref: &ast::NameRef| -> bool {
        let analyzer = SourceAnalyzer::new(db, file_id, name_ref.syntax(), None);
        let classified = classify_name_ref(db, file_id, &analyzer, &name_ref);
        if let Some(d) = classified {
            d == def
        } else {
            false
        }
    };

    for (file_id, text_range) in scope {
        let text = db.file_text(file_id);
        let parse = SourceFile::parse(&text);
        let syntax = parse.tree().syntax().clone();

        for (idx, _) in text.match_indices(pat) {
            let offset = TextUnit::from_usize(idx);
            if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(&syntax, offset) {
                let name_range = name_ref.syntax().text_range();

                if let Some(range) = text_range {
                    if name_range.is_subrange(&range) && is_match(file_id, &name_ref) {
                        refs.push(ReferenceDescriptor {
                            name: name_ref.text().to_string(),
                            range: name_ref.syntax().text_range(),
                        });
                    }
                } else if is_match(file_id, &name_ref) {
                    refs.push(ReferenceDescriptor {
                        name: name_ref.text().to_string(),
                        range: name_ref.syntax().text_range(),
                    });
                }
            }
        }
    }

    return refs;
}

impl Definition {
    pub fn scope(&self, db: &RootDatabase) -> SearchScope {
        let module_src = self.container.definition_source(db);
        let file_id = module_src.file_id.original_file(db);

        if let NameKind::Pat((def, _)) = self.item {
            let range = match def {
                DefWithBody::Function(f) => f.source(db).ast.syntax().text_range(),
                DefWithBody::Const(c) => c.source(db).ast.syntax().text_range(),
                DefWithBody::Static(s) => s.source(db).ast.syntax().text_range(),
            };
            return SearchScope { scope: vec![(file_id, Some(range))] };
        }

        if let Some(ref vis) = self.visibility {
            let source_root_id = db.file_source_root(file_id);
            let source_root = db.source_root(source_root_id);
            let mut files = source_root.walk().map(|id| (id.into(), None)).collect::<Vec<_>>();

            if vis.syntax().text() == "pub(crate)" {
                return SearchScope { scope: files };
            }
            if vis.syntax().text() == "pub" {
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

                return SearchScope { scope: files };
            }
            // FIXME: "pub(super)", "pub(in path)"
        }

        let range = match module_src.ast {
            ModuleSource::Module(m) => Some(m.syntax().text_range()),
            ModuleSource::SourceFile(_) => None,
        };
        SearchScope { scope: vec![(file_id, range)] }
    }
}
