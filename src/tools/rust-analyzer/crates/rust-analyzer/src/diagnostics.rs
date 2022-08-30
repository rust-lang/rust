//! Book keeping for keeping diagnostics easily in sync with the client.
pub(crate) mod to_proto;

use std::{mem, sync::Arc};

use ide::FileId;
use ide_db::FxHashMap;
use stdx::hash::{NoHashHashMap, NoHashHashSet};

use crate::lsp_ext;

pub(crate) type CheckFixes = Arc<NoHashHashMap<usize, NoHashHashMap<FileId, Vec<Fix>>>>;

#[derive(Debug, Default, Clone)]
pub struct DiagnosticsMapConfig {
    pub remap_prefix: FxHashMap<String, String>,
    pub warnings_as_info: Vec<String>,
    pub warnings_as_hint: Vec<String>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct DiagnosticCollection {
    // FIXME: should be NoHashHashMap<FileId, Vec<ra_id::Diagnostic>>
    pub(crate) native: NoHashHashMap<FileId, Vec<lsp_types::Diagnostic>>,
    // FIXME: should be Vec<flycheck::Diagnostic>
    pub(crate) check: NoHashHashMap<usize, NoHashHashMap<FileId, Vec<lsp_types::Diagnostic>>>,
    pub(crate) check_fixes: CheckFixes,
    changes: NoHashHashSet<FileId>,
}

#[derive(Debug, Clone)]
pub(crate) struct Fix {
    // Fixes may be triggerable from multiple ranges.
    pub(crate) ranges: Vec<lsp_types::Range>,
    pub(crate) action: lsp_ext::CodeAction,
}

impl DiagnosticCollection {
    pub(crate) fn clear_check(&mut self, flycheck_id: usize) {
        if let Some(it) = Arc::make_mut(&mut self.check_fixes).get_mut(&flycheck_id) {
            it.clear();
        }
        if let Some(it) = self.check.get_mut(&flycheck_id) {
            self.changes.extend(it.drain().map(|(key, _value)| key));
        }
    }

    pub(crate) fn clear_check_all(&mut self) {
        Arc::make_mut(&mut self.check_fixes).clear();
        self.changes
            .extend(self.check.values_mut().flat_map(|it| it.drain().map(|(key, _value)| key)))
    }

    pub(crate) fn clear_native_for(&mut self, file_id: FileId) {
        self.native.remove(&file_id);
        self.changes.insert(file_id);
    }

    pub(crate) fn add_check_diagnostic(
        &mut self,
        flycheck_id: usize,
        file_id: FileId,
        diagnostic: lsp_types::Diagnostic,
        fix: Option<Fix>,
    ) {
        let diagnostics = self.check.entry(flycheck_id).or_default().entry(file_id).or_default();
        for existing_diagnostic in diagnostics.iter() {
            if are_diagnostics_equal(existing_diagnostic, &diagnostic) {
                return;
            }
        }

        let check_fixes = Arc::make_mut(&mut self.check_fixes);
        check_fixes.entry(flycheck_id).or_default().entry(file_id).or_default().extend(fix);
        diagnostics.push(diagnostic);
        self.changes.insert(file_id);
    }

    pub(crate) fn set_native_diagnostics(
        &mut self,
        file_id: FileId,
        diagnostics: Vec<lsp_types::Diagnostic>,
    ) {
        if let Some(existing_diagnostics) = self.native.get(&file_id) {
            if existing_diagnostics.len() == diagnostics.len()
                && diagnostics
                    .iter()
                    .zip(existing_diagnostics)
                    .all(|(new, existing)| are_diagnostics_equal(new, existing))
            {
                return;
            }
        }

        self.native.insert(file_id, diagnostics);
        self.changes.insert(file_id);
    }

    pub(crate) fn diagnostics_for(
        &self,
        file_id: FileId,
    ) -> impl Iterator<Item = &lsp_types::Diagnostic> {
        let native = self.native.get(&file_id).into_iter().flatten();
        let check =
            self.check.values().filter_map(move |it| it.get(&file_id)).into_iter().flatten();
        native.chain(check)
    }

    pub(crate) fn take_changes(&mut self) -> Option<NoHashHashSet<FileId>> {
        if self.changes.is_empty() {
            return None;
        }
        Some(mem::take(&mut self.changes))
    }
}

fn are_diagnostics_equal(left: &lsp_types::Diagnostic, right: &lsp_types::Diagnostic) -> bool {
    left.source == right.source
        && left.severity == right.severity
        && left.range == right.range
        && left.message == right.message
}
