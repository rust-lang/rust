//! FIXME: write short doc here

use std::any::Any;

use hir_expand::{db::AstDatabase, name::Name, HirFileId, InFile};
use ra_syntax::{ast, AstNode, AstPtr, SyntaxNodePtr};
use stdx::format_to;

pub use hir_def::{diagnostics::UnresolvedModule, expr::MatchArm, path::Path};
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
        InFile::new(self.file, self.field.clone().into())
    }

    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

impl AstDiagnostic for NoSuchField {
    type AST = ast::RecordField;

    fn ast(&self, db: &dyn AstDatabase) -> Self::AST {
        let root = db.parse_or_expand(self.source().file_id).unwrap();
        let node = self.source().value.to_node(&root);
        ast::RecordField::cast(node).unwrap()
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
        let mut buf = String::from("Missing structure fields:\n");
        for field in &self.missed_fields {
            format_to!(buf, "- {}\n", field);
        }
        buf
    }
    fn source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.field_list.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

impl AstDiagnostic for MissingFields {
    type AST = ast::RecordFieldList;

    fn ast(&self, db: &dyn AstDatabase) -> Self::AST {
        let root = db.parse_or_expand(self.source().file_id).unwrap();
        let node = self.source().value.to_node(&root);
        ast::RecordFieldList::cast(node).unwrap()
    }
}

#[derive(Debug)]
pub struct MissingPatFields {
    pub file: HirFileId,
    pub field_list: AstPtr<ast::RecordFieldPatList>,
    pub missed_fields: Vec<Name>,
}

impl Diagnostic for MissingPatFields {
    fn message(&self) -> String {
        let mut buf = String::from("Missing structure fields:\n");
        for field in &self.missed_fields {
            format_to!(buf, "- {}\n", field);
        }
        buf
    }
    fn source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.field_list.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

#[derive(Debug)]
pub struct MissingMatchArms {
    pub file: HirFileId,
    pub match_expr: AstPtr<ast::Expr>,
    pub arms: AstPtr<ast::MatchArmList>,
}

impl Diagnostic for MissingMatchArms {
    fn message(&self) -> String {
        String::from("Missing match arm")
    }
    fn source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.match_expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
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
        InFile { file_id: self.file, value: self.expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

impl AstDiagnostic for MissingOkInTailExpr {
    type AST = ast::Expr;

    fn ast(&self, db: &dyn AstDatabase) -> Self::AST {
        let root = db.parse_or_expand(self.file).unwrap();
        let node = self.source().value.to_node(&root);
        ast::Expr::cast(node).unwrap()
    }
}

#[derive(Debug)]
pub struct BreakOutsideOfLoop {
    pub file: HirFileId,
    pub expr: AstPtr<ast::Expr>,
}

impl Diagnostic for BreakOutsideOfLoop {
    fn message(&self) -> String {
        "break outside of loop".to_string()
    }
    fn source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

impl AstDiagnostic for BreakOutsideOfLoop {
    type AST = ast::Expr;

    fn ast(&self, db: &dyn AstDatabase) -> Self::AST {
        let root = db.parse_or_expand(self.file).unwrap();
        let node = self.source().value.to_node(&root);
        ast::Expr::cast(node).unwrap()
    }
}

#[derive(Debug)]
pub struct MissingUnsafe {
    pub file: HirFileId,
    pub expr: AstPtr<ast::Expr>,
}

impl Diagnostic for MissingUnsafe {
    fn message(&self) -> String {
        format!("This operation is unsafe and requires an unsafe function or block")
    }
    fn source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

impl AstDiagnostic for MissingUnsafe {
    type AST = ast::Expr;

    fn ast(&self, db: &dyn AstDatabase) -> Self::AST {
        let root = db.parse_or_expand(self.source().file_id).unwrap();
        let node = self.source().value.to_node(&root);
        ast::Expr::cast(node).unwrap()
    }
}
