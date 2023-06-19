use rustc_errors::{DiagnosticArgValue, IntoDiagnosticArg};
use rustc_macros::Diagnostic;
use rustc_middle::ty::Ty;
use rustc_span::{Span, Symbol};
use std::borrow::Cow;

struct ExitCode(Option<i32>);

impl IntoDiagnosticArg for ExitCode {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        let ExitCode(exit_code) = self;
        match exit_code {
            Some(t) => t.into_diagnostic_arg(),
            None => DiagnosticArgValue::Str(Cow::Borrowed("<signal>")),
        }
    }
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_basic_integer, code = "E0511")]
pub(crate) struct InvalidMonomorphizationBasicInteger<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_invalid_float_vector, code = "E0511")]
pub(crate) struct InvalidMonomorphizationInvalidFloatVector<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub elem_ty: &'a str,
    pub vec_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_not_float, code = "E0511")]
pub(crate) struct InvalidMonomorphizationNotFloat<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_unrecognized, code = "E0511")]
pub(crate) struct InvalidMonomorphizationUnrecognized {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_expected_signed_unsigned, code = "E0511")]
pub(crate) struct InvalidMonomorphizationExpectedSignedUnsigned<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub elem_ty: Ty<'a>,
    pub vec_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_unsupported_element, code = "E0511")]
pub(crate) struct InvalidMonomorphizationUnsupportedElement<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub in_ty: Ty<'a>,
    pub elem_ty: Ty<'a>,
    pub ret_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_invalid_bitmask, code = "E0511")]
pub(crate) struct InvalidMonomorphizationInvalidBitmask<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub ty: Ty<'a>,
    pub expected_int_bits: u64,
    pub expected_bytes: u64,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_simd_shuffle, code = "E0511")]
pub(crate) struct InvalidMonomorphizationSimdShuffle<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_expected_simd, code = "E0511")]
pub(crate) struct InvalidMonomorphizationExpectedSimd<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub position: &'a str,
    pub found_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_mask_type, code = "E0511")]
pub(crate) struct InvalidMonomorphizationMaskType<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_return_length, code = "E0511")]
pub(crate) struct InvalidMonomorphizationReturnLength<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub in_len: u64,
    pub ret_ty: Ty<'a>,
    pub out_len: u64,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_return_length_input_type, code = "E0511")]
pub(crate) struct InvalidMonomorphizationReturnLengthInputType<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub in_len: u64,
    pub in_ty: Ty<'a>,
    pub ret_ty: Ty<'a>,
    pub out_len: u64,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_return_element, code = "E0511")]
pub(crate) struct InvalidMonomorphizationReturnElement<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub in_elem: Ty<'a>,
    pub in_ty: Ty<'a>,
    pub ret_ty: Ty<'a>,
    pub out_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_return_type, code = "E0511")]
pub(crate) struct InvalidMonomorphizationReturnType<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub in_elem: Ty<'a>,
    pub in_ty: Ty<'a>,
    pub ret_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_inserted_type, code = "E0511")]
pub(crate) struct InvalidMonomorphizationInsertedType<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub in_elem: Ty<'a>,
    pub in_ty: Ty<'a>,
    pub out_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_return_integer_type, code = "E0511")]
pub(crate) struct InvalidMonomorphizationReturnIntegerType<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub ret_ty: Ty<'a>,
    pub out_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_mismatched_lengths, code = "E0511")]
pub(crate) struct InvalidMonomorphizationMismatchedLengths {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub m_len: u64,
    pub v_len: u64,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_unsupported_cast, code = "E0511")]
pub(crate) struct InvalidMonomorphizationUnsupportedCast<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub in_ty: Ty<'a>,
    pub in_elem: Ty<'a>,
    pub ret_ty: Ty<'a>,
    pub out_elem: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_unsupported_operation, code = "E0511")]
pub(crate) struct InvalidMonomorphizationUnsupportedOperation<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub in_ty: Ty<'a>,
    pub in_elem: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_lto_not_supported)]
pub(crate) struct LTONotSupported;

#[derive(Diagnostic)]
#[diag(codegen_gcc_unwinding_inline_asm)]
pub(crate) struct UnwindingInlineAsm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_minimum_alignment)]
pub(crate) struct InvalidMinimumAlignment {
    pub err: String,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_tied_target_features)]
#[help]
pub(crate) struct TiedTargetFeatures {
    #[primary_span]
    pub span: Span,
    pub features: String,
}
