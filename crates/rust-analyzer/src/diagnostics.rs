//! Book keeping for keeping diagnostics easily in sync with the client.
pub(crate) mod to_proto;

use std::{collections::HashMap, sync::Arc};

use lsp_types::{Diagnostic, Range};
use ra_ide::FileId;

use crate::lsp_ext;

pub type CheckFixes = Arc<HashMap<FileId, Vec<Fix>>>;

#[derive(Debug, Default, Clone)]
pub struct DiagnosticsConfig {
    pub warnings_as_info: Vec<String>,
    pub warnings_as_hint: Vec<String>,
}

#[derive(Debug, Default, Clone)]
pub struct DiagnosticCollection {
    pub native: HashMap<FileId, Vec<Diagnostic>>,
    pub check: HashMap<FileId, Vec<Diagnostic>>,
    pub check_fixes: CheckFixes,
}

#[derive(Debug, Clone)]
pub struct Fix {
    pub range: Range,
    pub action: lsp_ext::CodeAction,
}

#[derive(Debug)]
pub enum DiagnosticTask {
    ClearCheck,
    AddCheck(FileId, Diagnostic, Vec<lsp_ext::CodeAction>),
    SetNative(FileId, Vec<Diagnostic>),
}

impl DiagnosticCollection {
    pub fn clear_check(&mut self) -> Vec<FileId> {
        Arc::make_mut(&mut self.check_fixes).clear();
        self.check.drain().map(|(key, _value)| key).collect()
    }

    pub fn add_check_diagnostic(
        &mut self,
        file_id: FileId,
        diagnostic: Diagnostic,
        fixes: Vec<lsp_ext::CodeAction>,
    ) {
        let diagnostics = self.check.entry(file_id).or_default();
        for existing_diagnostic in diagnostics.iter() {
            if are_diagnostics_equal(&existing_diagnostic, &diagnostic) {
                return;
            }
        }

        let check_fixes = Arc::make_mut(&mut self.check_fixes);
        check_fixes
            .entry(file_id)
            .or_default()
            .extend(fixes.into_iter().map(|action| Fix { range: diagnostic.range, action }));
        diagnostics.push(diagnostic);
    }

    pub fn set_native_diagnostics(&mut self, file_id: FileId, diagnostics: Vec<Diagnostic>) {
        self.native.insert(file_id, diagnostics);
    }

    pub fn diagnostics_for(&self, file_id: FileId) -> impl Iterator<Item = &Diagnostic> {
        let native = self.native.get(&file_id).into_iter().flatten();
        let check = self.check.get(&file_id).into_iter().flatten();
        native.chain(check)
    }

    pub fn handle_task(&mut self, task: DiagnosticTask) -> Vec<FileId> {
        match task {
            DiagnosticTask::ClearCheck => self.clear_check(),
            DiagnosticTask::AddCheck(file_id, diagnostic, fixes) => {
                self.add_check_diagnostic(file_id, diagnostic, fixes);
                vec![file_id]
            }
            DiagnosticTask::SetNative(file_id, diagnostics) => {
                self.set_native_diagnostics(file_id, diagnostics);
                vec![file_id]
            }
        }
    }
}

fn are_diagnostics_equal(left: &Diagnostic, right: &Diagnostic) -> bool {
    left.source == right.source
        && left.severity == right.severity
        && left.range == right.range
        && left.message == right.message
}
