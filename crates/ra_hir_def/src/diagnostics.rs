//! Diagnostics produced by `hir_def`.

use std::any::Any;

use hir_expand::diagnostics::{Diagnostic, DiagnosticWithFix};
use ra_syntax::{ast, AstPtr, SyntaxNodePtr};

use hir_expand::{HirFileId, InFile};

#[derive(Debug)]
pub struct UnresolvedModule {
    pub file: HirFileId,
    pub decl: AstPtr<ast::Module>,
    pub candidate: String,
}

impl Diagnostic for UnresolvedModule {
    fn message(&self) -> String {
        "unresolved module".to_string()
    }
    fn presentation(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.decl.clone().into())
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

impl DiagnosticWithFix for UnresolvedModule {
    type AST = ast::Module;
    fn fix_source(&self, db: &dyn hir_expand::db::AstDatabase) -> Option<Self::AST> {
        let root = db.parse_or_expand(self.file)?;
        Some(self.decl.to_node(&root))
    }
}
