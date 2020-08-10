use hir::{
    db::AstDatabase,
    diagnostics::{MissingFields, MissingOkInTailExpr, NoSuchField, UnresolvedModule},
};
use ra_syntax::ast;

// TODO kb
pub trait DiagnosticWithFix {
    type AST;
    fn fix_source(&self, db: &dyn AstDatabase) -> Option<Self::AST>;
}

impl DiagnosticWithFix for UnresolvedModule {
    type AST = ast::Module;
    fn fix_source(&self, db: &dyn AstDatabase) -> Option<Self::AST> {
        let root = db.parse_or_expand(self.file)?;
        Some(self.decl.to_node(&root))
    }
}

impl DiagnosticWithFix for NoSuchField {
    type AST = ast::RecordExprField;

    fn fix_source(&self, db: &dyn AstDatabase) -> Option<Self::AST> {
        let root = db.parse_or_expand(self.file)?;
        Some(self.field.to_node(&root))
    }
}

impl DiagnosticWithFix for MissingFields {
    type AST = ast::RecordExpr;

    fn fix_source(&self, db: &dyn AstDatabase) -> Option<Self::AST> {
        let root = db.parse_or_expand(self.file)?;
        Some(self.field_list_parent.to_node(&root))
    }
}

impl DiagnosticWithFix for MissingOkInTailExpr {
    type AST = ast::Expr;

    fn fix_source(&self, db: &dyn AstDatabase) -> Option<Self::AST> {
        let root = db.parse_or_expand(self.file)?;
        Some(self.expr.to_node(&root))
    }
}
