use hir::{
    self, Problem, source_binder
};
use ra_ide_api_light::{self, LocalEdit, Severity};
use ra_syntax::{
    algo::find_node_at_offset, ast::{self, NameOwner}, AstNode,
    SourceFile,
    TextRange,
};
use ra_db::SourceDatabase;

use crate::{
    CrateId, db, Diagnostic, FileId, FilePosition, FileSystemEdit,
    Query, SourceChange, SourceFileEdit,
    symbol_index::FileSymbol,
};

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
