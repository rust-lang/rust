use rustc_errors::fluent;
use rustc_errors::DiagnosticBuilder;
use rustc_session::SessionDiagnostic;

pub(crate) enum UnknownCTargetFeature {
    UnknownFeaturePrefix { feature: String },
    UnknownFeature { feature: String, rust_feature: Option<String> },
}

impl SessionDiagnostic<'_, ()> for UnknownCTargetFeature {
    fn into_diagnostic(
        self,
        sess: &'_ rustc_session::parse::ParseSess,
    ) -> DiagnosticBuilder<'_, ()> {
        match self {
            UnknownCTargetFeature::UnknownFeaturePrefix { feature } => {
                let mut diag = sess.struct_warn(fluent::codegen_llvm::unknown_ctarget_feature);
                diag.set_arg("feature", feature);
                diag.note(fluent::codegen_llvm::unknown_feature_prefix);
                diag
            }
            UnknownCTargetFeature::UnknownFeature { feature, rust_feature } => {
                let mut diag = sess.struct_warn(fluent::codegen_llvm::unknown_ctarget_feature);
                diag.set_arg("feature", feature);
                diag.note(fluent::codegen_llvm::unknown_feature);
                if let Some(rust_feature) = rust_feature {
                    diag.help(fluent::codegen_llvm::rust_feature);
                    diag.set_arg("rust_feature", rust_feature);
                } else {
                    diag.note(fluent::codegen_llvm::unknown_feature_fill_request);
                }
                diag
            }
        }
    }
}
