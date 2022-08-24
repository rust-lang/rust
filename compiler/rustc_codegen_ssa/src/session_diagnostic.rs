use rustc_errors::{fluent, DiagnosticArgValue, IntoDiagnosticArg};
use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use std::borrow::Cow;

#[derive(SessionDiagnostic)]
#[diag(codegen_ssa::error)]
pub struct DeserializeRlinkError {
    #[subdiagnostic]
    pub sub: DeserializeRlinkErrorSub,
}

#[derive(SessionSubdiagnostic)]
pub enum DeserializeRlinkErrorSub {
    #[note(codegen_ssa::wrong_file_type)]
    WrongFileType,

    #[note(codegen_ssa::empty_version_number)]
    EmptyVersionNumber,

    #[note(codegen_ssa::encoding_version_mismatch)]
    EncodingVersionMismatch { version_array: String, rlink_version: String },

    #[note(codegen_ssa::rustc_version_mismatch)]
    RustcVersionMismatch { rustc_version: String, current_version: String },
}

impl IntoDiagnosticArg for DeserializeRlinkErrorSub {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Borrowed(match self {
            DeserializeRlinkErrorSub::WrongFileType => fluent::codegen_ssa::wrong_file_type,
            DeserializeRlinkErrorSub::EmptyVersionNumber => {
                fluent::codegen_ssa::empty_version_number
            }
            DeserializeRlinkErrorSub::EncodingVersionMismatch { version_array, rlink_version } => {
                fluent::codegen_ssa::encoding_version_mismatch
            }
            DeserializeRlinkErrorSub::RustcVersionMismatch { rustc_version, current_version } => {
                fluent::codegen_ssa::rustc_version_mismatch
            }
        }))
    }
}
