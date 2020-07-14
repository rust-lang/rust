//! FIXME: write short doc here
mod expr;
mod match_check;
mod unsafe_check;

use std::any::Any;

use hir_def::DefWithBodyId;
use hir_expand::diagnostics::{AstDiagnostic, Diagnostic, DiagnosticSink};
use hir_expand::{db::AstDatabase, name::Name, HirFileId, InFile};
use ra_prof::profile;
use ra_syntax::{ast, AstNode, AstPtr, SyntaxNodePtr};
use stdx::format_to;

use crate::db::HirDatabase;

pub use crate::diagnostics::expr::{record_literal_missing_fields, record_pattern_missing_fields};

pub fn validate_body(db: &dyn HirDatabase, owner: DefWithBodyId, sink: &mut DiagnosticSink<'_>) {
    let _p = profile("validate_body");
    let infer = db.infer(owner);
    infer.add_diagnostics(db, owner, sink);
    let mut validator = expr::ExprValidator::new(owner, infer.clone(), sink);
    validator.validate_body(db);
    let mut validator = unsafe_check::UnsafeValidator::new(owner, infer, sink);
    validator.validate_body(db);
}

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

#[derive(Debug)]
pub struct MismatchedArgCount {
    pub file: HirFileId,
    pub call_expr: AstPtr<ast::Expr>,
    pub expected: usize,
    pub found: usize,
}

impl Diagnostic for MismatchedArgCount {
    fn message(&self) -> String {
        let s = if self.expected == 1 { "" } else { "s" };
        format!("Expected {} argument{}, found {}", self.expected, s, self.found)
    }
    fn source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.call_expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

impl AstDiagnostic for MismatchedArgCount {
    type AST = ast::CallExpr;
    fn ast(&self, db: &dyn AstDatabase) -> Self::AST {
        let root = db.parse_or_expand(self.source().file_id).unwrap();
        let node = self.source().value.to_node(&root);
        ast::CallExpr::cast(node).unwrap()
    }
}

#[cfg(test)]
fn check_diagnostics(ra_fixture: &str) {
    use ra_db::{fixture::WithFixture, FileId};
    use ra_syntax::TextRange;
    use rustc_hash::FxHashMap;

    use crate::test_db::TestDB;

    let db = TestDB::with_files(ra_fixture);
    let annotations = db.extract_annotations();

    let mut actual: FxHashMap<FileId, Vec<(TextRange, String)>> = FxHashMap::default();
    db.diag(|d| {
        // FXIME: macros...
        let file_id = d.source().file_id.original_file(&db);
        let range = d.syntax_node(&db).text_range();
        let message = d.message().to_owned();
        actual.entry(file_id).or_default().push((range, message));
    });
    actual.values_mut().for_each(|diags| diags.sort_by_key(|it| it.0.start()));

    assert_eq!(annotations, actual);
}
