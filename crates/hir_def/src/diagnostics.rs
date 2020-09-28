//! Diagnostics produced by `hir_def`.

use std::any::Any;

use hir_expand::diagnostics::{Diagnostic, DiagnosticCode};
use syntax::{ast, AstPtr, SyntaxNodePtr};

use hir_expand::{HirFileId, InFile};

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
