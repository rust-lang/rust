use crate::{fluent, DiagnosticBuilder, Handler, IntoDiagnostic};
use rustc_target::abi::TargetDataLayoutErrors;

impl IntoDiagnostic<'_, !> for TargetDataLayoutErrors<'_> {
    fn into_diagnostic(self, handler: &Handler) -> DiagnosticBuilder<'_, !> {
        let mut diag;
        match self {
            TargetDataLayoutErrors::InvalidAddressSpace { addr_space, err, cause } => {
                diag = handler.struct_fatal(fluent::errors::target_invalid_address_space);
                diag.set_arg("addr_space", addr_space);
                diag.set_arg("cause", cause);
                diag.set_arg("err", err);
                diag
            }
            TargetDataLayoutErrors::InvalidBits { kind, bit, cause, err } => {
                diag = handler.struct_fatal(fluent::errors::target_invalid_bits);
                diag.set_arg("kind", kind);
                diag.set_arg("bit", bit);
                diag.set_arg("cause", cause);
                diag.set_arg("err", err);
                diag
            }
            TargetDataLayoutErrors::MissingAlignment { cause } => {
                diag = handler.struct_fatal(fluent::errors::target_missing_alignment);
                diag.set_arg("cause", cause);
                diag
            }
            TargetDataLayoutErrors::InvalidAlignment { cause, err } => {
                diag = handler.struct_fatal(fluent::errors::target_invalid_alignment);
                diag.set_arg("cause", cause);
                diag.set_arg("err", err);
                diag
            }
            TargetDataLayoutErrors::InconsistentTargetArchitecture { dl, target } => {
                diag = handler.struct_fatal(fluent::errors::target_inconsistent_architecture);
                diag.set_arg("dl", dl);
                diag.set_arg("target", target);
                diag
            }
            TargetDataLayoutErrors::InconsistentTargetPointerWidth { pointer_size, target } => {
                diag = handler.struct_fatal(fluent::errors::target_inconsistent_pointer_width);
                diag.set_arg("pointer_size", pointer_size);
                diag.set_arg("target", target);
                diag
            }
            TargetDataLayoutErrors::InvalidBitsSize { err } => {
                diag = handler.struct_fatal(fluent::errors::target_invalid_bits_size);
                diag.set_arg("err", err);
                diag
            }
        }
    }
}
