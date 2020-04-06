//! FIXME: write short doc here

use std::sync::Arc;

use hir_def::{
    path::{path, Path},
    resolver::HasResolver,
    AdtId, FunctionId,
};
use hir_expand::{diagnostics::DiagnosticSink, name::Name};
use ra_syntax::ast;
use ra_syntax::AstPtr;
use rustc_hash::FxHashSet;

use crate::{
    db::HirDatabase,
    diagnostics::{MissingFields, MissingMatchArms, MissingOkInTailExpr},
    utils::variant_data,
    ApplicationTy, InferenceResult, Ty, TypeCtor,
    _match::{is_useful, MatchCheckCtx, Matrix, PatStack, Usefulness},
};

pub use hir_def::{
    body::{
        scope::{ExprScopes, ScopeEntry, ScopeId},
        Body, BodySourceMap, ExprPtr, ExprSource, PatPtr, PatSource,
    },
    expr::{
        ArithOp, Array, BinaryOp, BindingAnnotation, CmpOp, Expr, ExprId, Literal, LogicOp,
        MatchArm, Ordering, Pat, PatId, RecordFieldPat, RecordLitField, Statement, UnaryOp,
    },
    VariantId,
};

pub struct ExprValidator<'a, 'b: 'a> {
    func: FunctionId,
    infer: Arc<InferenceResult>,
    sink: &'a mut DiagnosticSink<'b>,
}

impl<'a, 'b> ExprValidator<'a, 'b> {
    pub fn new(
        func: FunctionId,
        infer: Arc<InferenceResult>,
        sink: &'a mut DiagnosticSink<'b>,
    ) -> ExprValidator<'a, 'b> {
        ExprValidator { func, infer, sink }
    }

    pub fn validate_body(&mut self, db: &dyn HirDatabase) {
        let body = db.body(self.func.into());

        for e in body.exprs.iter() {
            if let (id, Expr::RecordLit { path, fields, spread }) = e {
                self.validate_record_literal(id, path, fields, *spread, db);
            } else if let (id, Expr::Match { expr, arms }) = e {
                self.validate_match(id, *expr, arms, db, self.infer.clone());
            }
        }

        let body_expr = &body[body.body_expr];
        if let Expr::Block { tail: Some(t), .. } = body_expr {
            self.validate_results_in_tail_expr(body.body_expr, *t, db);
        }
    }

    fn validate_match(
        &mut self,
        id: ExprId,
        match_expr: ExprId,
        arms: &[MatchArm],
        db: &dyn HirDatabase,
        infer: Arc<InferenceResult>,
    ) {
        let (body, source_map): (Arc<Body>, Arc<BodySourceMap>) =
            db.body_with_source_map(self.func.into());

        let match_expr_ty = match infer.type_of_expr.get(match_expr) {
            Some(ty) => ty,
            // If we can't resolve the type of the match expression
            // we cannot perform exhaustiveness checks.
            None => return,
        };

        let cx = MatchCheckCtx { body, infer: infer.clone(), db };
        let pats = arms.iter().map(|arm| arm.pat);

        let mut seen = Matrix::empty();
        for pat in pats {
            // We skip any patterns whose type we cannot resolve.
            if let Some(pat_ty) = infer.type_of_pat.get(pat) {
                // We only include patterns whose type matches the type
                // of the match expression. If we had a InvalidMatchArmPattern
                // diagnostic or similar we could raise that in an else
                // block here.
                //
                // When comparing the types, we also have to consider that rustc
                // will automatically de-reference the match expression type if
                // necessary.
                if pat_ty == match_expr_ty
                    || match_expr_ty
                        .as_reference()
                        .map(|(match_expr_ty, _)| match_expr_ty == pat_ty)
                        .unwrap_or(false)
                {
                    // If we had a NotUsefulMatchArm diagnostic, we could
                    // check the usefulness of each pattern as we added it
                    // to the matrix here.
                    let v = PatStack::from_pattern(pat);
                    seen.push(&cx, v);
                }
            }
        }

        match is_useful(&cx, &seen, &PatStack::from_wild()) {
            Ok(Usefulness::Useful) => (),
            // if a wildcard pattern is not useful, then all patterns are covered
            Ok(Usefulness::NotUseful) => return,
            // this path is for unimplemented checks, so we err on the side of not
            // reporting any errors
            _ => return,
        }

        if let Ok(source_ptr) = source_map.expr_syntax(id) {
            if let Some(expr) = source_ptr.value.left() {
                let root = source_ptr.file_syntax(db.upcast());
                if let ast::Expr::MatchExpr(match_expr) = expr.to_node(&root) {
                    if let (Some(match_expr), Some(arms)) =
                        (match_expr.expr(), match_expr.match_arm_list())
                    {
                        self.sink.push(MissingMatchArms {
                            file: source_ptr.file_id,
                            match_expr: AstPtr::new(&match_expr),
                            arms: AstPtr::new(&arms),
                        })
                    }
                }
            }
        }
    }

    fn validate_record_literal(
        &mut self,
        id: ExprId,
        _path: &Option<Path>,
        fields: &[RecordLitField],
        spread: Option<ExprId>,
        db: &dyn HirDatabase,
    ) {
        if spread.is_some() {
            return;
        };
        let variant_def: VariantId = match self.infer.variant_resolution_for_expr(id) {
            Some(VariantId::UnionId(_)) | None => return,
            Some(it) => it,
        };
        if let VariantId::UnionId(_) = variant_def {
            return;
        }

        let variant_data = variant_data(db.upcast(), variant_def);

        let lit_fields: FxHashSet<_> = fields.iter().map(|f| &f.name).collect();
        let missed_fields: Vec<Name> = variant_data
            .fields()
            .iter()
            .filter_map(|(_f, d)| {
                let name = d.name.clone();
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
        let (_, source_map) = db.body_with_source_map(self.func.into());

        if let Ok(source_ptr) = source_map.expr_syntax(id) {
            if let Some(expr) = source_ptr.value.left() {
                let root = source_ptr.file_syntax(db.upcast());
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

    fn validate_results_in_tail_expr(&mut self, body_id: ExprId, id: ExprId, db: &dyn HirDatabase) {
        // the mismatch will be on the whole block currently
        let mismatch = match self.infer.type_mismatch_for_expr(body_id) {
            Some(m) => m,
            None => return,
        };

        let std_result_path = path![std::result::Result];

        let resolver = self.func.resolver(db.upcast());
        let std_result_enum = match resolver.resolve_known_enum(db.upcast(), &std_result_path) {
            Some(it) => it,
            _ => return,
        };

        let std_result_ctor = TypeCtor::Adt(AdtId::EnumId(std_result_enum));
        let params = match &mismatch.expected {
            Ty::Apply(ApplicationTy { ctor, parameters }) if ctor == &std_result_ctor => parameters,
            _ => return,
        };

        if params.len() == 2 && params[0] == mismatch.actual {
            let (_, source_map) = db.body_with_source_map(self.func.into());

            if let Ok(source_ptr) = source_map.expr_syntax(id) {
                if let Some(expr) = source_ptr.value.left() {
                    self.sink.push(MissingOkInTailExpr { file: source_ptr.file_id, expr });
                }
            }
        }
    }
}
