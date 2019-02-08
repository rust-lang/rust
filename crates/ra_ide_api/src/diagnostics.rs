use hir::{Problem, source_binder};
use ra_ide_api_light::Severity;
use ra_db::SourceDatabase;

use crate::{db, Diagnostic, FileId, FileSystemEdit, SourceChange};

impl db::RootDatabase {
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
}
