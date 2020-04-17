//! Diagnostics produced by `hir_def`.

use std::any::Any;

use hir_expand::diagnostics::Diagnostic;
use ra_db::RelativePathBuf;
use ra_syntax::{ast, AstPtr, SyntaxNodePtr, TextRange};

use hir_expand::{HirFileId, InFile};

#[derive(Debug)]
pub struct UnresolvedModule {
    pub file: HirFileId,
    pub decl: AstPtr<ast::Module>,
    pub highlight_range: TextRange,
    pub candidate: RelativePathBuf,
}

impl Diagnostic for UnresolvedModule {
    fn message(&self) -> String {
        "unresolved module".to_string()
    }
    fn highlight_range(&self) -> InFile<TextRange> {
        InFile::new(self.file, self.highlight_range)
    }
    fn source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.decl.clone().into())
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}
