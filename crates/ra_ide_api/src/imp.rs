use std::{
    sync::Arc,
    time,
};

use hir::{
    self, Problem, source_binder
};
use ra_db::{
    SourceDatabase, SourceRoot, SourceRootId,
    salsa::{Database, SweepStrategy},
};
use ra_ide_api_light::{self, LocalEdit, Severity};
use ra_syntax::{
    algo::find_node_at_offset, ast::{self, NameOwner}, AstNode,
    SourceFile,
    TextRange,
};

use crate::{
    AnalysisChange,
    CrateId, db, Diagnostic, FileId, FilePosition, FileSystemEdit,
    Query, RootChange, SourceChange, SourceFileEdit,
    symbol_index::{FileSymbol, SymbolsDatabase},
    status::syntax_tree_stats
};

const GC_COOLDOWN: time::Duration = time::Duration::from_millis(100);

impl db::RootDatabase {
    pub(crate) fn apply_change(&mut self, change: AnalysisChange) {
        log::info!("apply_change {:?}", change);
        if !change.new_roots.is_empty() {
            let mut local_roots = Vec::clone(&self.local_roots());
            for (root_id, is_local) in change.new_roots {
                self.set_source_root(root_id, Default::default());
                if is_local {
                    local_roots.push(root_id);
                }
            }
            self.set_local_roots(Arc::new(local_roots));
        }

        for (root_id, root_change) in change.roots_changed {
            self.apply_root_change(root_id, root_change);
        }
        for (file_id, text) in change.files_changed {
            self.set_file_text(file_id, text)
        }
        if !change.libraries_added.is_empty() {
            let mut libraries = Vec::clone(&self.library_roots());
            for library in change.libraries_added {
                libraries.push(library.root_id);
                self.set_source_root(library.root_id, Default::default());
                self.set_constant_library_symbols(library.root_id, Arc::new(library.symbol_index));
                self.apply_root_change(library.root_id, library.root_change);
            }
            self.set_library_roots(Arc::new(libraries));
        }
        if let Some(crate_graph) = change.crate_graph {
            self.set_crate_graph(Arc::new(crate_graph))
        }
    }

    fn apply_root_change(&mut self, root_id: SourceRootId, root_change: RootChange) {
        let mut source_root = SourceRoot::clone(&self.source_root(root_id));
        for add_file in root_change.added {
            self.set_file_text(add_file.file_id, add_file.text);
            self.set_file_relative_path(add_file.file_id, add_file.path.clone());
            self.set_file_source_root(add_file.file_id, root_id);
            source_root.files.insert(add_file.path, add_file.file_id);
        }
        for remove_file in root_change.removed {
            self.set_file_text(remove_file.file_id, Default::default());
            source_root.files.remove(&remove_file.path);
        }
        self.set_source_root(root_id, Arc::new(source_root));
    }

    pub(crate) fn maybe_collect_garbage(&mut self) {
        if self.last_gc_check.elapsed() > GC_COOLDOWN {
            self.last_gc_check = time::Instant::now();
            let retained_trees = syntax_tree_stats(self).retained;
            if retained_trees > 100 {
                log::info!(
                    "automatic garbadge collection, {} retained trees",
                    retained_trees
                );
                self.collect_garbage();
            }
        }
    }

    pub(crate) fn collect_garbage(&mut self) {
        self.last_gc = time::Instant::now();

        let sweep = SweepStrategy::default()
            .discard_values()
            .sweep_all_revisions();

        self.query(ra_db::ParseQuery).sweep(sweep);

        self.query(hir::db::HirParseQuery).sweep(sweep);
        self.query(hir::db::FileItemsQuery).sweep(sweep);
        self.query(hir::db::FileItemQuery).sweep(sweep);

        self.query(hir::db::LowerModuleQuery).sweep(sweep);
        self.query(hir::db::LowerModuleSourceMapQuery).sweep(sweep);
        self.query(hir::db::BodySyntaxMappingQuery).sweep(sweep);
    }
}

impl db::RootDatabase {
    /// Returns `Vec` for the same reason as `parent_module`
    pub(crate) fn crate_for(&self, file_id: FileId) -> Vec<CrateId> {
        let module = match source_binder::module_from_file_id(self, file_id) {
            Some(it) => it,
            None => return Vec::new(),
        };
        let krate = match module.krate(self) {
            Some(it) => it,
            None => return Vec::new(),
        };
        vec![krate.crate_id()]
    }

    pub(crate) fn find_all_refs(&self, position: FilePosition) -> Vec<(FileId, TextRange)> {
        let file = self.parse(position.file_id);
        // Find the binding associated with the offset
        let (binding, descr) = match find_binding(self, &file, position) {
            None => return Vec::new(),
            Some(it) => it,
        };

        let mut ret = binding
            .name()
            .into_iter()
            .map(|name| (position.file_id, name.syntax().range()))
            .collect::<Vec<_>>();
        ret.extend(
            descr
                .scopes(self)
                .find_all_refs(binding)
                .into_iter()
                .map(|ref_desc| (position.file_id, ref_desc.range)),
        );

        return ret;

        fn find_binding<'a>(
            db: &db::RootDatabase,
            source_file: &'a SourceFile,
            position: FilePosition,
        ) -> Option<(&'a ast::BindPat, hir::Function)> {
            let syntax = source_file.syntax();
            if let Some(binding) = find_node_at_offset::<ast::BindPat>(syntax, position.offset) {
                let descr = source_binder::function_from_child_node(
                    db,
                    position.file_id,
                    binding.syntax(),
                )?;
                return Some((binding, descr));
            };
            let name_ref = find_node_at_offset::<ast::NameRef>(syntax, position.offset)?;
            let descr =
                source_binder::function_from_child_node(db, position.file_id, name_ref.syntax())?;
            let scope = descr.scopes(db);
            let resolved = scope.resolve_local_name(name_ref)?;
            let resolved = resolved.ptr().to_node(source_file);
            let binding = find_node_at_offset::<ast::BindPat>(syntax, resolved.range().end())?;
            Some((binding, descr))
        }
    }

    pub(crate) fn diagnostics(&self, file_id: FileId) -> Vec<Diagnostic> {
        let syntax = self.parse(file_id);

        let mut res = ra_ide_api_light::diagnostics(&syntax)
            .into_iter()
            .map(|d| Diagnostic {
                range: d.range,
                message: d.msg,
                severity: d.severity,
                fix: d.fix.map(|fix| SourceChange::from_local_edit(file_id, fix)),
            })
            .collect::<Vec<_>>();
        if let Some(m) = source_binder::module_from_file_id(self, file_id) {
            for (name_node, problem) in m.problems(self) {
                let source_root = self.file_source_root(file_id);
                let diag = match problem {
                    Problem::UnresolvedModule { candidate } => {
                        let create_file = FileSystemEdit::CreateFile {
                            source_root,
                            path: candidate.clone(),
                        };
                        let fix = SourceChange {
                            label: "create module".to_string(),
                            source_file_edits: Vec::new(),
                            file_system_edits: vec![create_file],
                            cursor_position: None,
                        };
                        Diagnostic {
                            range: name_node.range(),
                            message: "unresolved module".to_string(),
                            severity: Severity::Error,
                            fix: Some(fix),
                        }
                    }
                    Problem::NotDirOwner { move_to, candidate } => {
                        let move_file = FileSystemEdit::MoveFile {
                            src: file_id,
                            dst_source_root: source_root,
                            dst_path: move_to.clone(),
                        };
                        let create_file = FileSystemEdit::CreateFile {
                            source_root,
                            path: move_to.join(candidate),
                        };
                        let fix = SourceChange {
                            label: "move file and create module".to_string(),
                            source_file_edits: Vec::new(),
                            file_system_edits: vec![move_file, create_file],
                            cursor_position: None,
                        };
                        Diagnostic {
                            range: name_node.range(),
                            message: "can't declare module at this location".to_string(),
                            severity: Severity::Error,
                            fix: Some(fix),
                        }
                    }
                };
                res.push(diag)
            }
        };
        res
    }

    pub(crate) fn index_resolve(&self, name_ref: &ast::NameRef) -> Vec<FileSymbol> {
        let name = name_ref.text();
        let mut query = Query::new(name.to_string());
        query.exact();
        query.limit(4);
        crate::symbol_index::world_symbols(self, query)
    }
}

impl SourceChange {
    pub(crate) fn from_local_edit(file_id: FileId, edit: LocalEdit) -> SourceChange {
        let file_edit = SourceFileEdit {
            file_id,
            edit: edit.edit,
        };
        SourceChange {
            label: edit.label,
            source_file_edits: vec![file_edit],
            file_system_edits: vec![],
            cursor_position: edit
                .cursor_position
                .map(|offset| FilePosition { offset, file_id }),
        }
    }
}
