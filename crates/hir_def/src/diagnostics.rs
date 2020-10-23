//! Diagnostics produced by `hir_def`.

use std::any::Any;
use stdx::format_to;

use cfg::{CfgExpr, CfgOptions, DnfExpr};
use hir_expand::diagnostics::{Diagnostic, DiagnosticCode, DiagnosticSink};
use hir_expand::{HirFileId, InFile};
use syntax::{ast, AstPtr, SyntaxNodePtr};

use crate::{db::DefDatabase, DefWithBodyId};

pub fn validate_body(db: &dyn DefDatabase, owner: DefWithBodyId, sink: &mut DiagnosticSink<'_>) {
    let source_map = db.body_with_source_map(owner).1;
    source_map.add_diagnostics(db, sink);
}

// Diagnostic: unresolved-module
//
// This diagnostic is triggered if rust-analyzer is unable to discover referred module.
#[derive(Debug)]
pub struct UnresolvedModule {
    pub file: HirFileId,
    pub decl: AstPtr<ast::Module>,
    pub candidate: String,
}

impl Diagnostic for UnresolvedModule {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("unresolved-module")
    }
    fn message(&self) -> String {
        "unresolved module".to_string()
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.decl.clone().into())
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: unresolved-extern-crate
//
// This diagnostic is triggered if rust-analyzer is unable to discover referred extern crate.
#[derive(Debug)]
pub struct UnresolvedExternCrate {
    pub file: HirFileId,
    pub item: AstPtr<ast::ExternCrate>,
}

impl Diagnostic for UnresolvedExternCrate {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("unresolved-extern-crate")
    }
    fn message(&self) -> String {
        "unresolved extern crate".to_string()
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.item.clone().into())
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: unresolved-import
//
// This diagnostic is triggered if rust-analyzer is unable to discover imported module.
#[derive(Debug)]
pub struct UnresolvedImport {
    pub file: HirFileId,
    pub node: AstPtr<ast::UseTree>,
}

impl Diagnostic for UnresolvedImport {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("unresolved-import")
    }
    fn message(&self) -> String {
        "unresolved import".to_string()
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.node.clone().into())
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
    fn is_experimental(&self) -> bool {
        // This currently results in false positives in the following cases:
        // - `cfg_if!`-generated code in libstd (we don't load the sysroot correctly)
        // - `core::arch` (we don't handle `#[path = "../<path>"]` correctly)
        // - proc macros and/or proc macro generated code
        true
    }
}

// Diagnostic: unconfigured-code
//
// This diagnostic is shown for code with inactive `#[cfg]` attributes.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct InactiveCode {
    pub file: HirFileId,
    pub node: SyntaxNodePtr,
    pub cfg: CfgExpr,
    pub opts: CfgOptions,
}

impl Diagnostic for InactiveCode {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("inactive-code")
    }
    fn message(&self) -> String {
        let inactive = DnfExpr::new(self.cfg.clone()).why_inactive(&self.opts);
        let mut buf = "code is inactive due to #[cfg] directives".to_string();

        if let Some(inactive) = inactive {
            format_to!(buf, ": {}", inactive);
        }

        buf
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.node.clone())
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}
