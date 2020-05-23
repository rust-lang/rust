//! FIXME: write short doc here
pub use hir_def::diagnostics::UnresolvedModule;
pub use hir_expand::diagnostics::{AstDiagnostic, Diagnostic, DiagnosticSink};
pub use hir_ty::diagnostics::{MissingFields, MissingMatchArms, MissingOkInTailExpr, NoSuchField};

use std::sync::Arc;

use crate::code_model::Function;
use crate::db::HirDatabase;
use crate::has_source::HasSource;
use hir_ty::{
    diagnostics::{MissingUnsafe, UnnecessaryUnsafe},
    expr::unsafe_expressions,
    InferenceResult,
};
use ra_syntax::AstPtr;

pub struct UnsafeValidator<'a, 'b: 'a> {
    func: &'a Function,
    infer: Arc<InferenceResult>,
    sink: &'a mut DiagnosticSink<'b>,
}

impl<'a, 'b> UnsafeValidator<'a, 'b> {
    pub fn new(
        func: &'a Function,
        infer: Arc<InferenceResult>,
        sink: &'a mut DiagnosticSink<'b>,
    ) -> UnsafeValidator<'a, 'b> {
        UnsafeValidator { func, infer, sink }
    }

    pub fn validate_body(&mut self, db: &dyn HirDatabase) {
        let def = self.func.id.into();
        let unsafe_expressions = unsafe_expressions(db, self.infer.as_ref(), def);
        let func_data = db.function_data(self.func.id);
        let unnecessary = func_data.is_unsafe && unsafe_expressions.len() == 0;
        let missing = !func_data.is_unsafe && unsafe_expressions.len() > 0;
        if !(unnecessary || missing) {
            return;
        }

        let in_file = self.func.source(db);
        let file = in_file.file_id;
        let fn_def = AstPtr::new(&in_file.value);
        let fn_name = func_data.name.clone().into();

        if unnecessary {
            self.sink.push(UnnecessaryUnsafe { file, fn_def, fn_name })
        } else {
            self.sink.push(MissingUnsafe { file, fn_def, fn_name })
        }
    }
}
