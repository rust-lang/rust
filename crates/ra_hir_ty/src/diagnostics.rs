//! FIXME: write short doc here

use std::any::Any;

use hir_expand::{db::AstDatabase, name::Name, HirFileId, InFile};
use ra_syntax::{ast, AstNode, AstPtr, SyntaxNodePtr};

pub use hir_def::diagnostics::UnresolvedModule;
pub use hir_expand::diagnostics::{AstDiagnostic, Diagnostic, DiagnosticSink};

#[derive(Debug)]
pub struct NoSuchField {
    pub file: HirFileId,
    pub field: AstPtr<ast::RecordField>,
}

impl Diagnostic for NoSuchField {
    fn message(&self) -> String {
        "no such field".to_string()
    }

    fn source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.field.into() }
    }

    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

#[derive(Debug)]
pub struct MissingFields {
    pub file: HirFileId,
    pub field_list: AstPtr<ast::RecordFieldList>,
    pub missed_fields: Vec<Name>,
}

impl Diagnostic for MissingFields {
    fn message(&self) -> String {
        use std::fmt::Write;
        let mut message = String::from("Missing structure fields:\n");
        for field in &self.missed_fields {
            write!(message, "- {}\n", field).unwrap();
        }
        message
    }
    fn source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.field_list.into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

impl AstDiagnostic for MissingFields {
    type AST = ast::RecordFieldList;

    fn ast(&self, db: &impl AstDatabase) -> Self::AST {
        let root = db.parse_or_expand(self.source().file_id).unwrap();
        let node = self.source().value.to_node(&root);
        ast::RecordFieldList::cast(node).unwrap()
    }
}

#[derive(Debug)]
pub struct MissingOkInTailExpr {
    pub file: HirFileId,
    pub expr: AstPtr<ast::Expr>,
}

impl Diagnostic for MissingOkInTailExpr {
    fn message(&self) -> String {
        "wrap return expression in Ok".to_string()
    }
    fn source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.expr.into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

impl AstDiagnostic for MissingOkInTailExpr {
    type AST = ast::Expr;

    fn ast(&self, db: &impl AstDatabase) -> Self::AST {
        let root = db.parse_or_expand(self.file).unwrap();
        let node = self.source().value.to_node(&root);
        ast::Expr::cast(node).unwrap()
    }
}
