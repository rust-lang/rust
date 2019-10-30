//! FIXME: write short doc here

use std::sync::Arc;

use hir_def::path::known;
use ra_syntax::ast;
use rustc_hash::FxHashSet;

use crate::{
    db::HirDatabase,
    diagnostics::{DiagnosticSink, MissingFields, MissingOkInTailExpr},
    expr::AstPtr,
    ty::{ApplicationTy, InferenceResult, Ty, TypeCtor},
    Adt, Function, Name, Path,
};

use super::{Expr, ExprId, RecordLitField};

pub(crate) struct ExprValidator<'a, 'b: 'a> {
    func: Function,
    infer: Arc<InferenceResult>,
    sink: &'a mut DiagnosticSink<'b>,
}

impl<'a, 'b> ExprValidator<'a, 'b> {
    pub(crate) fn new(
        func: Function,
        infer: Arc<InferenceResult>,
        sink: &'a mut DiagnosticSink<'b>,
    ) -> ExprValidator<'a, 'b> {
        ExprValidator { func, infer, sink }
    }

    pub(crate) fn validate_body(&mut self, db: &impl HirDatabase) {
        let body = self.func.body(db);

        for e in body.exprs() {
            if let (id, Expr::RecordLit { path, fields, spread }) = e {
                self.validate_record_literal(id, path, fields, *spread, db);
            }
        }

        let body_expr = &body[body.body_expr()];
        if let Expr::Block { statements: _, tail: Some(t) } = body_expr {
            self.validate_results_in_tail_expr(body.body_expr(), *t, db);
        }
    }

    fn validate_record_literal(
        &mut self,
        id: ExprId,
        _path: &Option<Path>,
        fields: &[RecordLitField],
        spread: Option<ExprId>,
        db: &impl HirDatabase,
    ) {
        if spread.is_some() {
            return;
        }

        let struct_def = match self.infer[id].as_adt() {
            Some((Adt::Struct(s), _)) => s,
            _ => return,
        };

        let lit_fields: FxHashSet<_> = fields.iter().map(|f| &f.name).collect();
        let missed_fields: Vec<Name> = struct_def
            .fields(db)
            .iter()
            .filter_map(|f| {
                let name = f.name(db);
                if lit_fields.contains(&name) {
                    None
                } else {
                    Some(name)
                }
            })
            .collect();
        if missed_fields.is_empty() {
            return;
        }
        let source_map = self.func.body_source_map(db);

        if let Some(source_ptr) = source_map.expr_syntax(id) {
            if let Some(expr) = source_ptr.ast.a() {
                let root = source_ptr.file_syntax(db);
                if let ast::Expr::RecordLit(record_lit) = expr.to_node(&root) {
                    if let Some(field_list) = record_lit.record_field_list() {
                        self.sink.push(MissingFields {
                            file: source_ptr.file_id,
                            field_list: AstPtr::new(&field_list),
                            missed_fields,
                        })
                    }
                }
            }
        }
    }

    fn validate_results_in_tail_expr(
        &mut self,
        body_id: ExprId,
        id: ExprId,
        db: &impl HirDatabase,
    ) {
        // the mismatch will be on the whole block currently
        let mismatch = match self.infer.type_mismatch_for_expr(body_id) {
            Some(m) => m,
            None => return,
        };

        let std_result_path = known::std_result_result();

        let resolver = self.func.resolver(db);
        let std_result_enum = match resolver.resolve_known_enum(db, &std_result_path) {
            Some(it) => it,
            _ => return,
        };

        let std_result_ctor = TypeCtor::Adt(Adt::Enum(std_result_enum));
        let params = match &mismatch.expected {
            Ty::Apply(ApplicationTy { ctor, parameters }) if ctor == &std_result_ctor => parameters,
            _ => return,
        };

        if params.len() == 2 && &params[0] == &mismatch.actual {
            let source_map = self.func.body_source_map(db);

            if let Some(source_ptr) = source_map.expr_syntax(id) {
                if let Some(expr) = source_ptr.ast.a() {
                    self.sink.push(MissingOkInTailExpr { file: source_ptr.file_id, expr });
                }
            }
        }
    }
}
