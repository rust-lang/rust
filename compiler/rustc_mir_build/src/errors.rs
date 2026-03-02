use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level,
    MultiSpan, Subdiagnostic, msg,
};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_middle::ty::{self, Ty};
use rustc_pattern_analysis::errors::Uncovered;
use rustc_pattern_analysis::rustc::RustcPatCtxt;
use rustc_span::{Ident, Span, Symbol};

#[derive(Diagnostic)]
#[diag("call to deprecated safe function `{$function}` is unsafe and requires unsafe block")]
pub(crate) struct CallToDeprecatedSafeFnRequiresUnsafe {
    #[label("call to unsafe function")]
    pub(crate) span: Span,
    pub(crate) function: String,
    #[subdiagnostic]
    pub(crate) sub: CallToDeprecatedSafeFnRequiresUnsafeSub,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "you can wrap the call in an `unsafe` block if you can guarantee {$guarantee}",
    applicability = "machine-applicable"
)]
pub(crate) struct CallToDeprecatedSafeFnRequiresUnsafeSub {
    pub(crate) start_of_line_suggestion: String,
    #[suggestion_part(code = "{start_of_line_suggestion}")]
    pub(crate) start_of_line: Span,
    #[suggestion_part(code = "unsafe {{ ")]
    pub(crate) left: Span,
    #[suggestion_part(code = " }}")]
    pub(crate) right: Span,
    pub(crate) guarantee: String,
}

#[derive(Diagnostic)]
#[diag("call to unsafe function `{$function}` is unsafe and requires unsafe block", code = E0133)]
#[note("consult the function's documentation for information on how to avoid undefined behavior")]
pub(crate) struct UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafe {
    #[label("call to unsafe function")]
    pub(crate) span: Span,
    pub(crate) function: String,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag("call to unsafe function is unsafe and requires unsafe block", code = E0133)]
#[note("consult the function's documentation for information on how to avoid undefined behavior")]
pub(crate) struct UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafeNameless {
    #[label("call to unsafe function")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag("use of inline assembly is unsafe and requires unsafe block", code = E0133)]
#[note("inline assembly is entirely unchecked and can cause undefined behavior")]
pub(crate) struct UnsafeOpInUnsafeFnUseOfInlineAssemblyRequiresUnsafe {
    #[label("use of inline assembly")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag("initializing type with `rustc_layout_scalar_valid_range` attr is unsafe and requires unsafe block", code = E0133)]
#[note(
    "initializing a layout restricted type's field with a value outside the valid range is undefined behavior"
)]
pub(crate) struct UnsafeOpInUnsafeFnInitializingTypeWithRequiresUnsafe {
    #[label("initializing type with `rustc_layout_scalar_valid_range` attr")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag("initializing type with an unsafe field is unsafe and requires unsafe block", code = E0133)]
#[note("unsafe fields may carry library invariants")]
pub(crate) struct UnsafeOpInUnsafeFnInitializingTypeWithUnsafeFieldRequiresUnsafe {
    #[label("initialization of struct with unsafe field")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag("use of mutable static is unsafe and requires unsafe block", code = E0133)]
#[note(
    "mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior"
)]
pub(crate) struct UnsafeOpInUnsafeFnUseOfMutableStaticRequiresUnsafe {
    #[label("use of mutable static")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag("use of extern static is unsafe and requires unsafe block", code = E0133)]
#[note(
    "extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior"
)]
pub(crate) struct UnsafeOpInUnsafeFnUseOfExternStaticRequiresUnsafe {
    #[label("use of extern static")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag("use of unsafe field is unsafe and requires unsafe block", code = E0133)]
#[note("unsafe fields may carry library invariants")]
pub(crate) struct UnsafeOpInUnsafeFnUseOfUnsafeFieldRequiresUnsafe {
    #[label("use of unsafe field")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag("dereference of raw pointer is unsafe and requires unsafe block", code = E0133)]
#[note(
    "raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior"
)]
pub(crate) struct UnsafeOpInUnsafeFnDerefOfRawPointerRequiresUnsafe {
    #[label("dereference of raw pointer")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag("access to union field is unsafe and requires unsafe block", code = E0133)]
#[note(
    "the field may not be properly initialized: using uninitialized data will cause undefined behavior"
)]
pub(crate) struct UnsafeOpInUnsafeFnAccessToUnionFieldRequiresUnsafe {
    #[label("access to union field")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag(
    "mutation of layout constrained field is unsafe and requires unsafe block",
    code = E0133
)]
#[note("mutating layout constrained fields cannot statically be checked for valid values")]
pub(crate) struct UnsafeOpInUnsafeFnMutationOfLayoutConstrainedFieldRequiresUnsafe {
    #[label("mutation of layout constrained field")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag(
    "borrow of layout constrained field with interior mutability is unsafe and requires unsafe block",
    code = E0133,
)]
pub(crate) struct UnsafeOpInUnsafeFnBorrowOfLayoutConstrainedFieldRequiresUnsafe {
    #[label("borrow of layout constrained field with interior mutability")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag(
    "unsafe binder cast is unsafe and requires unsafe block information that may be required to uphold safety guarantees of a type",
    code = E0133,
)]
pub(crate) struct UnsafeOpInUnsafeFnUnsafeBinderCastRequiresUnsafe {
    #[label("unsafe binder cast")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag("call to function `{$function}` with `#[target_feature]` is unsafe and requires unsafe block", code = E0133)]
#[help(
    "in order for the call to be safe, the context requires the following additional target {$missing_target_features_count ->
    [1] feature
    *[count] features
    }: {$missing_target_features}"
)]
pub(crate) struct UnsafeOpInUnsafeFnCallToFunctionWithRequiresUnsafe {
    #[label("call to function with `#[target_feature]`")]
    pub(crate) span: Span,
    pub(crate) function: String,
    pub(crate) missing_target_features: DiagArgValue,
    pub(crate) missing_target_features_count: usize,
    #[note("the {$build_target_features} target {$build_target_features_count ->
        [1] feature
        *[count] features
    } being enabled in the build configuration does not remove the requirement to list {$build_target_features_count ->
        [1] it
        *[count] them
    } in `#[target_feature]`")]
    pub(crate) note: bool,
    pub(crate) build_target_features: DiagArgValue,
    pub(crate) build_target_features_count: usize,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag("call to unsafe function `{$function}` is unsafe and requires unsafe block", code = E0133)]
#[note("consult the function's documentation for information on how to avoid undefined behavior")]
pub(crate) struct CallToUnsafeFunctionRequiresUnsafe {
    #[primary_span]
    #[label("call to unsafe function")]
    pub(crate) span: Span,
    pub(crate) function: String,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("call to unsafe function is unsafe and requires unsafe block", code = E0133)]
#[note("consult the function's documentation for information on how to avoid undefined behavior")]
pub(crate) struct CallToUnsafeFunctionRequiresUnsafeNameless {
    #[primary_span]
    #[label("call to unsafe function")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("call to unsafe function `{$function}` is unsafe and requires unsafe function or block", code = E0133)]
#[note("consult the function's documentation for information on how to avoid undefined behavior")]
pub(crate) struct CallToUnsafeFunctionRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("call to unsafe function")]
    pub(crate) span: Span,
    pub(crate) function: String,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    "call to unsafe function is unsafe and requires unsafe function or block",
    code = E0133
)]
#[note("consult the function's documentation for information on how to avoid undefined behavior")]
pub(crate) struct CallToUnsafeFunctionRequiresUnsafeNamelessUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("call to unsafe function")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("use of inline assembly is unsafe and requires unsafe block", code = E0133)]
#[note("inline assembly is entirely unchecked and can cause undefined behavior")]
pub(crate) struct UseOfInlineAssemblyRequiresUnsafe {
    #[primary_span]
    #[label("use of inline assembly")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("use of inline assembly is unsafe and requires unsafe function or block", code = E0133)]
#[note("inline assembly is entirely unchecked and can cause undefined behavior")]
pub(crate) struct UseOfInlineAssemblyRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("use of inline assembly")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("initializing type with `rustc_layout_scalar_valid_range` attr is unsafe and requires unsafe block", code = E0133)]
#[note(
    "initializing a layout restricted type's field with a value outside the valid range is undefined behavior"
)]
pub(crate) struct InitializingTypeWithRequiresUnsafe {
    #[primary_span]
    #[label("initializing type with `rustc_layout_scalar_valid_range` attr")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("initializing type with an unsafe field is unsafe and requires unsafe block", code = E0133)]
#[note("unsafe fields may carry library invariants")]
pub(crate) struct InitializingTypeWithUnsafeFieldRequiresUnsafe {
    #[primary_span]
    #[label("initialization of struct with unsafe field")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    "initializing type with `rustc_layout_scalar_valid_range` attr is unsafe and requires unsafe function or block",
    code = E0133
)]
#[note(
    "initializing a layout restricted type's field with a value outside the valid range is undefined behavior"
)]
pub(crate) struct InitializingTypeWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("initializing type with `rustc_layout_scalar_valid_range` attr")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    "initializing type with an unsafe field is unsafe and requires unsafe block",
    code = E0133
)]
#[note("unsafe fields may carry library invariants")]
pub(crate) struct InitializingTypeWithUnsafeFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("initialization of struct with unsafe field")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("use of mutable static is unsafe and requires unsafe block", code = E0133)]
#[note(
    "mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior"
)]
pub(crate) struct UseOfMutableStaticRequiresUnsafe {
    #[primary_span]
    #[label("use of mutable static")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("use of mutable static is unsafe and requires unsafe function or block", code = E0133)]
#[note(
    "mutable statics can be mutated by multiple threads: aliasing violations or data races will cause undefined behavior"
)]
pub(crate) struct UseOfMutableStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("use of mutable static")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("use of extern static is unsafe and requires unsafe block", code = E0133)]
#[note(
    "extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior"
)]
pub(crate) struct UseOfExternStaticRequiresUnsafe {
    #[primary_span]
    #[label("use of extern static")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("use of extern static is unsafe and requires unsafe function or block", code = E0133)]
#[note(
    "extern statics are not controlled by the Rust type system: invalid data, aliasing violations or data races will cause undefined behavior"
)]
pub(crate) struct UseOfExternStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("use of extern static")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("use of unsafe field is unsafe and requires unsafe block", code = E0133)]
#[note("unsafe fields may carry library invariants")]
pub(crate) struct UseOfUnsafeFieldRequiresUnsafe {
    #[primary_span]
    #[label("use of unsafe field")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("use of unsafe field is unsafe and requires unsafe block", code = E0133)]
#[note("unsafe fields may carry library invariants")]
pub(crate) struct UseOfUnsafeFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("use of unsafe field")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("dereference of raw pointer is unsafe and requires unsafe block", code = E0133)]
#[note(
    "raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior"
)]
pub(crate) struct DerefOfRawPointerRequiresUnsafe {
    #[primary_span]
    #[label("dereference of raw pointer")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("dereference of raw pointer is unsafe and requires unsafe function or block", code = E0133)]
#[note(
    "raw pointers may be null, dangling or unaligned; they can violate aliasing rules and cause data races: all of these are undefined behavior"
)]
pub(crate) struct DerefOfRawPointerRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("dereference of raw pointer")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("access to union field is unsafe and requires unsafe block", code = E0133)]
#[note(
    "the field may not be properly initialized: using uninitialized data will cause undefined behavior"
)]
pub(crate) struct AccessToUnionFieldRequiresUnsafe {
    #[primary_span]
    #[label("access to union field")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("access to union field is unsafe and requires unsafe function or block", code = E0133)]
#[note(
    "the field may not be properly initialized: using uninitialized data will cause undefined behavior"
)]
pub(crate) struct AccessToUnionFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("access to union field")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("mutation of layout constrained field is unsafe and requires unsafe block", code = E0133)]
#[note("mutating layout constrained fields cannot statically be checked for valid values")]
pub(crate) struct MutationOfLayoutConstrainedFieldRequiresUnsafe {
    #[primary_span]
    #[label("mutation of layout constrained field")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    "mutation of layout constrained field is unsafe and requires unsafe function or block",
    code = E0133
)]
#[note("mutating layout constrained fields cannot statically be checked for valid values")]
pub(crate) struct MutationOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("mutation of layout constrained field")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("borrow of layout constrained field with interior mutability is unsafe and requires unsafe block", code = E0133)]
#[note(
    "references to fields of layout constrained fields lose the constraints. Coupled with interior mutability, the field can be changed to invalid values"
)]
pub(crate) struct BorrowOfLayoutConstrainedFieldRequiresUnsafe {
    #[primary_span]
    #[label("borrow of layout constrained field with interior mutability")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    "borrow of layout constrained field with interior mutability is unsafe and requires unsafe function or block",
    code = E0133
)]
#[note(
    "references to fields of layout constrained fields lose the constraints. Coupled with interior mutability, the field can be changed to invalid values"
)]
pub(crate) struct BorrowOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("borrow of layout constrained field with interior mutability")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag("call to function `{$function}` with `#[target_feature]` is unsafe and requires unsafe block", code = E0133)]
#[help(
    "in order for the call to be safe, the context requires the following additional target {$missing_target_features_count ->
    [1] feature
    *[count] features
}: {$missing_target_features}"
)]
pub(crate) struct CallToFunctionWithRequiresUnsafe {
    #[primary_span]
    #[label("call to function with `#[target_feature]`")]
    pub(crate) span: Span,
    pub(crate) function: String,
    pub(crate) missing_target_features: DiagArgValue,
    pub(crate) missing_target_features_count: usize,
    #[note("the {$build_target_features} target {$build_target_features_count ->
        [1] feature
        *[count] features
    } being enabled in the build configuration does not remove the requirement to list {$build_target_features_count ->
        [1] it
        *[count] them
    } in `#[target_feature]`")]
    pub(crate) note: bool,
    pub(crate) build_target_features: DiagArgValue,
    pub(crate) build_target_features_count: usize,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    "call to function `{$function}` with `#[target_feature]` is unsafe and requires unsafe function or block",
    code = E0133,
)]
#[help(
    "in order for the call to be safe, the context requires the following additional target {$missing_target_features_count ->
    [1] feature
    *[count] features
}: {$missing_target_features}"
)]
pub(crate) struct CallToFunctionWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("call to function with `#[target_feature]`")]
    pub(crate) span: Span,
    pub(crate) function: String,
    pub(crate) missing_target_features: DiagArgValue,
    pub(crate) missing_target_features_count: usize,
    #[note("the {$build_target_features} target {$build_target_features_count ->
    [1] feature
    *[count] features
    } being enabled in the build configuration does not remove the requirement to list {$build_target_features_count ->
    [1] it
    *[count] them
    } in `#[target_feature]`")]
    pub(crate) note: bool,
    pub(crate) build_target_features: DiagArgValue,
    pub(crate) build_target_features_count: usize,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    "unsafe binder cast is unsafe and requires unsafe block information that may be required to uphold safety guarantees of a type",
    code = E0133,
)]
pub(crate) struct UnsafeBinderCastRequiresUnsafe {
    #[primary_span]
    #[label("unsafe binder cast")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    "unsafe binder cast is unsafe and requires unsafe block or unsafe fn information that may be required to uphold safety guarantees of a type",
    code = E0133,
)]
pub(crate) struct UnsafeBinderCastRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label("unsafe binder cast")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Subdiagnostic)]
#[label("items do not inherit unsafety from separate enclosing items")]
pub(crate) struct UnsafeNotInheritedNote {
    #[primary_span]
    pub(crate) span: Span,
}

pub(crate) struct UnsafeNotInheritedLintNote {
    pub(crate) signature_span: Span,
    pub(crate) body_span: Span,
}

impl Subdiagnostic for UnsafeNotInheritedLintNote {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.span_note(
            self.signature_span,
            msg!("an unsafe function restricts its caller, but its body is safe by default"),
        );
        let body_start = self.body_span.shrink_to_lo();
        let body_end = self.body_span.shrink_to_hi();
        diag.tool_only_multipart_suggestion(
            msg!("consider wrapping the function body in an unsafe block"),
            vec![(body_start, "{ unsafe ".into()), (body_end, "}".into())],
            Applicability::MachineApplicable,
        );
    }
}

#[derive(Diagnostic)]
#[diag("unnecessary `unsafe` block")]
pub(crate) struct UnusedUnsafe {
    #[label("unnecessary `unsafe` block")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) enclosing: Option<UnusedUnsafeEnclosing>,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnusedUnsafeEnclosing {
    #[label("because it's nested under this `unsafe` block")]
    Block {
        #[primary_span]
        span: Span,
    },
}

pub(crate) struct NonExhaustivePatternsTypeNotEmpty<'p, 'tcx, 'm> {
    pub(crate) cx: &'m RustcPatCtxt<'p, 'tcx>,
    pub(crate) scrut_span: Span,
    pub(crate) braces_span: Option<Span>,
    pub(crate) ty: Ty<'tcx>,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for NonExhaustivePatternsTypeNotEmpty<'_, '_, '_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let mut diag =
            Diag::new(dcx, level, msg!("non-exhaustive patterns: type `{$ty}` is non-empty"));
        diag.span(self.scrut_span);
        diag.code(E0004);
        let peeled_ty = self.ty.peel_refs();
        diag.arg("ty", self.ty);
        diag.arg("peeled_ty", peeled_ty);

        if let ty::Adt(def, _) = peeled_ty.kind() {
            let def_span = self
                .cx
                .tcx
                .hir_get_if_local(def.did())
                .and_then(|node| node.ident())
                .map(|ident| ident.span)
                .unwrap_or_else(|| self.cx.tcx.def_span(def.did()));

            // workaround to make test pass
            let mut span: MultiSpan = def_span.into();
            span.push_span_label(def_span, "");

            diag.span_note(span, msg!("`{$peeled_ty}` defined here"));
        }

        let is_non_exhaustive = matches!(self.ty.kind(),
            ty::Adt(def, _) if def.variant_list_has_applicable_non_exhaustive());
        if is_non_exhaustive {
            diag.note(msg!(
                "the matched value is of type `{$ty}`, which is marked as non-exhaustive"
            ));
        } else {
            diag.note(msg!("the matched value is of type `{$ty}`"));
        }

        if let ty::Ref(_, sub_ty, _) = self.ty.kind() {
            if !sub_ty.is_inhabited_from(self.cx.tcx, self.cx.module, self.cx.typing_env) {
                diag.note(msg!("references are always considered inhabited"));
            }
        }

        let sm = self.cx.tcx.sess.source_map();
        if let Some(braces_span) = self.braces_span {
            // Get the span for the empty match body `{}`.
            let (indentation, more) = if let Some(snippet) = sm.indentation_before(self.scrut_span)
            {
                (format!("\n{snippet}"), "    ")
            } else {
                (" ".to_string(), "")
            };
            diag.span_suggestion_verbose(
                braces_span,
                msg!("ensure that all possible cases are being handled by adding a match arm with a wildcard pattern as shown"),
                format!(" {{{indentation}{more}_ => todo!(),{indentation}}}"),
                Applicability::HasPlaceholders,
            );
        } else {
            diag.help(msg!(
                "ensure that all possible cases are being handled by adding a match arm with a wildcard pattern"
            ));
        }

        diag
    }
}

#[derive(Subdiagnostic)]
#[note("match arms with guards don't count towards exhaustivity")]
pub(crate) struct NonExhaustiveMatchAllArmsGuarded;

#[derive(Diagnostic)]
#[diag("statics cannot be referenced in patterns", code = E0158)]
pub(crate) struct StaticInPattern {
    #[primary_span]
    #[label("can't be used in patterns")]
    pub(crate) span: Span,
    #[label("`static` defined here")]
    pub(crate) static_span: Span,
}

#[derive(Diagnostic)]
#[diag("constant parameters cannot be referenced in patterns", code = E0158)]
pub(crate) struct ConstParamInPattern {
    #[primary_span]
    #[label("can't be used in patterns")]
    pub(crate) span: Span,
    #[label("constant defined here")]
    pub(crate) const_span: Span,
}

#[derive(Diagnostic)]
#[diag("runtime values cannot be referenced in patterns", code = E0080)]
pub(crate) struct NonConstPath {
    #[primary_span]
    #[label("references a runtime value")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("unreachable pattern")]
pub(crate) struct UnreachablePattern<'tcx> {
    #[label("no value can reach this")]
    pub(crate) span: Option<Span>,
    #[label("matches no values because `{$matches_no_values_ty}` is uninhabited")]
    pub(crate) matches_no_values: Option<Span>,
    pub(crate) matches_no_values_ty: Ty<'tcx>,
    #[note(
        "to learn more about uninhabited types, see https://doc.rust-lang.org/nomicon/exotic-sizes.html#empty-types"
    )]
    pub(crate) uninhabited_note: Option<()>,
    #[label("matches any value")]
    pub(crate) covered_by_catchall: Option<Span>,
    #[subdiagnostic]
    pub(crate) wanted_constant: Option<WantedConstant>,
    #[note(
        "there is a constant of the same name imported in another scope, which could have been used to pattern match against its value instead of introducing a new catch-all binding, but it needs to be imported in the pattern's scope"
    )]
    pub(crate) accessible_constant: Option<Span>,
    #[note(
        "there is a constant of the same name, which could have been used to pattern match against its value instead of introducing a new catch-all binding, but it is not accessible from this scope"
    )]
    pub(crate) inaccessible_constant: Option<Span>,
    #[note(
        "there is a binding of the same name; if you meant to pattern match against the value of that binding, that is a feature of constants that is not available for `let` bindings"
    )]
    pub(crate) pattern_let_binding: Option<Span>,
    #[label("matches all the relevant values")]
    pub(crate) covered_by_one: Option<Span>,
    #[note("multiple earlier patterns match some of the same values")]
    pub(crate) covered_by_many: Option<MultiSpan>,
    pub(crate) covered_by_many_n_more_count: usize,
    #[suggestion("remove the match arm", code = "", applicability = "machine-applicable")]
    pub(crate) suggest_remove: Option<Span>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "you might have meant to pattern match against the value of {$is_typo ->
    [true] similarly named constant
    *[false] constant
    } `{$const_name}` instead of introducing a new catch-all binding",
    code = "{const_path}",
    applicability = "machine-applicable"
)]
pub(crate) struct WantedConstant {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) is_typo: bool,
    pub(crate) const_name: String,
    pub(crate) const_path: String,
}

#[derive(Diagnostic)]
#[diag("unreachable {$descr}")]
pub(crate) struct UnreachableDueToUninhabited<'desc, 'tcx> {
    pub descr: &'desc str,
    #[label("unreachable {$descr}")]
    pub expr: Span,
    #[label("any code following this expression is unreachable")]
    #[note("this expression has type `{$ty}`, which is uninhabited")]
    pub orig: Span,
    pub ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("constant pattern cannot depend on generic parameters", code = E0158)]
pub(crate) struct ConstPatternDependsOnGenericParameter {
    #[primary_span]
    #[label("`const` depends on a generic parameter")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("could not evaluate constant pattern")]
pub(crate) struct CouldNotEvalConstPattern {
    #[primary_span]
    #[label("could not evaluate constant")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("lower bound for range pattern must be less than or equal to upper bound", code = E0030)]
pub(crate) struct LowerRangeBoundMustBeLessThanOrEqualToUpper {
    #[primary_span]
    #[label("lower bound larger than upper bound")]
    pub(crate) span: Span,
    #[note(
        "when matching against a range, the compiler verifies that the range is non-empty. Range patterns include both end-points, so this is equivalent to requiring the start of the range to be less than or equal to the end of the range"
    )]
    pub(crate) teach: bool,
}

#[derive(Diagnostic)]
#[diag("literal out of range for `{$ty}`")]
pub(crate) struct LiteralOutOfRange<'tcx> {
    #[primary_span]
    #[label("this value does not fit into the type `{$ty}` whose range is `{$min}..={$max}`")]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
    pub(crate) min: i128,
    pub(crate) max: u128,
}

#[derive(Diagnostic)]
#[diag("lower bound for range pattern must be less than upper bound", code = E0579)]
pub(crate) struct LowerRangeBoundMustBeLessThanUpper {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("exclusive upper bound for a range bound cannot be the minimum", code = E0579)]
pub(crate) struct UpperRangeBoundCannotBeMin {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("pattern binding `{$name}` is named the same as one of the variants of the type `{$ty_path}`", code = E0170)]
pub(crate) struct BindingsWithVariantName {
    #[suggestion(
        "to match on the variant, qualify the path",
        code = "{ty_path}::{name}",
        applicability = "machine-applicable"
    )]
    pub(crate) suggestion: Option<Span>,
    pub(crate) ty_path: String,
    pub(crate) name: Ident,
}

#[derive(Diagnostic)]
#[diag(
    "irrefutable `if let` {$count ->
    [one] pattern
    *[other] patterns
}"
)]
#[note(
    "{$count ->
    [one] this pattern
    *[other] these patterns
} will always match, so the `if let` is useless"
)]
#[help("consider replacing the `if let` with a `let`")]
pub(crate) struct IrrefutableLetPatternsIfLet {
    pub(crate) count: usize,
}

#[derive(Diagnostic)]
#[diag(
    "irrefutable `if let` guard {$count ->
    [one] pattern
    *[other] patterns
}"
)]
#[note(
    "{$count ->
    [one] this pattern
    *[other] these patterns
} will always match, so the guard is useless"
)]
#[help("consider removing the guard and adding a `let` inside the match arm")]
pub(crate) struct IrrefutableLetPatternsIfLetGuard {
    pub(crate) count: usize,
}

#[derive(Diagnostic)]
#[diag(
    "irrefutable `let...else` {$count ->
    [one] pattern
    *[other] patterns
}"
)]
#[note(
    "{$count ->
    [one] this pattern
    *[other] these patterns
} will always match, so the `else` clause is useless"
)]
#[help("consider removing the `else` clause")]
pub(crate) struct IrrefutableLetPatternsLetElse {
    pub(crate) count: usize,
}

#[derive(Diagnostic)]
#[diag(
    "irrefutable `while let` {$count ->
    [one] pattern
    *[other] patterns
}"
)]
#[note(
    "{$count ->
    [one] this pattern
    *[other] these patterns
} will always match, so the loop will never exit"
)]
#[help("consider instead using a `loop {\"{\"} ... {\"}\"}` with a `let` inside it")]
pub(crate) struct IrrefutableLetPatternsWhileLet {
    pub(crate) count: usize,
}

#[derive(Diagnostic)]
#[diag("borrow of moved value")]
pub(crate) struct BorrowOfMovedValue<'tcx> {
    #[primary_span]
    #[label("value moved into `{$name}` here")]
    #[label(
        "move occurs because `{$name}` has type `{$ty}`, which does not implement the `Copy` trait"
    )]
    pub(crate) binding_span: Span,
    #[label("value borrowed here after move")]
    pub(crate) conflicts_ref: Vec<Span>,
    pub(crate) name: Ident,
    pub(crate) ty: Ty<'tcx>,
    #[suggestion(
        "borrow this binding in the pattern to avoid moving the value",
        code = "ref ",
        applicability = "machine-applicable"
    )]
    pub(crate) suggest_borrowing: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("cannot borrow value as mutable more than once at a time")]
pub(crate) struct MultipleMutBorrows {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) occurrences: Vec<Conflict>,
}

#[derive(Diagnostic)]
#[diag("cannot borrow value as mutable because it is also borrowed as immutable")]
pub(crate) struct AlreadyBorrowed {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) occurrences: Vec<Conflict>,
}

#[derive(Diagnostic)]
#[diag("cannot borrow value as immutable because it is also borrowed as mutable")]
pub(crate) struct AlreadyMutBorrowed {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) occurrences: Vec<Conflict>,
}

#[derive(Diagnostic)]
#[diag("cannot move out of value because it is borrowed")]
pub(crate) struct MovedWhileBorrowed {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) occurrences: Vec<Conflict>,
}

#[derive(Subdiagnostic)]
pub(crate) enum Conflict {
    #[label("value is mutably borrowed by `{$name}` here")]
    Mut {
        #[primary_span]
        span: Span,
        name: Symbol,
    },
    #[label("value is borrowed by `{$name}` here")]
    Ref {
        #[primary_span]
        span: Span,
        name: Symbol,
    },
    #[label("value is moved into `{$name}` here")]
    Moved {
        #[primary_span]
        span: Span,
        name: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag("cannot use unions in constant patterns")]
pub(crate) struct UnionPattern {
    #[primary_span]
    #[label("can't use a `union` here")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("constant of non-structural type `{$ty}` in a pattern")]
pub(crate) struct TypeNotStructural<'tcx> {
    #[primary_span]
    #[label("constant of non-structural type")]
    pub(crate) span: Span,
    #[label("`{$ty}` must be annotated with `#[derive(PartialEq)]` to be usable in patterns")]
    pub(crate) ty_def_span: Span,
    pub(crate) ty: Ty<'tcx>,
    #[note(
        "the `PartialEq` trait must be derived, manual `impl`s are not sufficient; see https://doc.rust-lang.org/stable/std/marker/trait.StructuralPartialEq.html for details"
    )]
    pub(crate) manual_partialeq_impl_span: Option<Span>,
    #[note(
        "see https://doc.rust-lang.org/stable/std/marker/trait.StructuralPartialEq.html for details"
    )]
    pub(crate) manual_partialeq_impl_note: bool,
}

#[derive(Diagnostic)]
#[diag("constant of non-structural type `{$ty}` in a pattern")]
#[note(
    "see https://doc.rust-lang.org/stable/std/marker/trait.StructuralPartialEq.html for details"
)]
pub(crate) struct TypeNotPartialEq<'tcx> {
    #[primary_span]
    #[label("constant of non-structural type")]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("{$prefix} `{$non_sm_ty}` cannot be used in patterns")]
pub(crate) struct InvalidPattern<'tcx> {
    #[primary_span]
    #[label("{$prefix} can't be used in patterns")]
    pub(crate) span: Span,
    pub(crate) non_sm_ty: Ty<'tcx>,
    pub(crate) prefix: String,
}

#[derive(Diagnostic)]
#[diag("cannot use unsized non-slice type `{$non_sm_ty}` in constant patterns")]
pub(crate) struct UnsizedPattern<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) non_sm_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("cannot use NaN in patterns")]
#[note("NaNs compare inequal to everything, even themselves, so this pattern would never match")]
#[help("try using the `is_nan` method instead")]
pub(crate) struct NaNPattern {
    #[primary_span]
    #[label("evaluates to `NaN`, which is not allowed in patterns")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "function pointers and raw pointers not derived from integers in patterns behave unpredictably and should not be relied upon"
)]
#[note("see https://github.com/rust-lang/rust/issues/70861 for details")]
pub(crate) struct PointerPattern {
    #[primary_span]
    #[label("can't be used in patterns")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("mismatched types")]
#[note("the matched value is of type `{$ty}`")]
pub(crate) struct NonEmptyNeverPattern<'tcx> {
    #[primary_span]
    #[label("a never pattern must be used on an uninhabited type")]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("refutable pattern in {$origin}", code = E0005)]
pub(crate) struct PatternNotCovered<'s, 'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) origin: &'s str,
    #[subdiagnostic]
    pub(crate) uncovered: Uncovered,
    #[subdiagnostic]
    pub(crate) inform: Option<Inform>,
    #[subdiagnostic]
    pub(crate) interpreted_as_const: Option<InterpretedAsConst>,
    #[subdiagnostic]
    pub(crate) interpreted_as_const_sugg: Option<InterpretedAsConstSugg>,
    #[subdiagnostic]
    pub(crate) adt_defined_here: Option<AdtDefinedHere<'tcx>>,
    #[note(
        "pattern `{$witness_1}` is currently uninhabited, but this variant contains private fields which may become inhabited in the future"
    )]
    pub(crate) witness_1_is_privately_uninhabited: bool,
    pub(crate) witness_1: String,
    #[note("the matched value is of type `{$pattern_ty}`")]
    pub(crate) _p: (),
    pub(crate) pattern_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub(crate) let_suggestion: Option<SuggestLet>,
    #[subdiagnostic]
    pub(crate) misc_suggestion: Option<MiscPatternSuggestion>,
}

#[derive(Subdiagnostic)]
#[note(
    "`let` bindings require an \"irrefutable pattern\", like a `struct` or an `enum` with only one variant"
)]
#[note("for more information, visit https://doc.rust-lang.org/book/ch19-02-refutability.html")]
pub(crate) struct Inform;

#[derive(Subdiagnostic)]
#[label(
    "missing patterns are not covered because `{$variable}` is interpreted as a constant pattern, not a new variable"
)]
pub(crate) struct InterpretedAsConst {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) variable: String,
}

pub(crate) struct AdtDefinedHere<'tcx> {
    pub(crate) adt_def_span: Span,
    pub(crate) ty: Ty<'tcx>,
    pub(crate) variants: Vec<Variant>,
}

pub(crate) struct Variant {
    pub(crate) span: Span,
}

impl<'tcx> Subdiagnostic for AdtDefinedHere<'tcx> {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.arg("ty", self.ty);
        let mut spans = MultiSpan::from(self.adt_def_span);

        for Variant { span } in self.variants {
            spans.push_span_label(span, msg!("not covered"));
        }

        diag.span_note(spans, msg!("`{$ty}` defined here"));
    }
}

#[derive(Subdiagnostic)]
#[suggestion(
    "introduce a variable instead",
    code = "{variable}_var",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct InterpretedAsConstSugg {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) variable: String,
}

#[derive(Subdiagnostic)]
pub(crate) enum SuggestLet {
    #[multipart_suggestion(
        "you might want to use `if let` to ignore the {$count ->
            [one] variant that isn't
            *[other] variants that aren't
        } matched",
        applicability = "has-placeholders"
    )]
    If {
        #[suggestion_part(code = "if ")]
        start_span: Span,
        #[suggestion_part(code = " {{ todo!() }}")]
        semi_span: Span,
        count: usize,
    },
    #[suggestion(
        "you might want to use `let...else` to handle the {$count ->
            [one] variant that isn't
            *[other] variants that aren't
        } matched",
        code = " else {{ todo!() }}",
        applicability = "has-placeholders"
    )]
    Else {
        #[primary_span]
        end_span: Span,
        count: usize,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum MiscPatternSuggestion {
    #[suggestion(
        "alternatively, you could prepend the pattern with an underscore to define a new named variable; identifiers cannot begin with digits",
        code = "_",
        applicability = "maybe-incorrect"
    )]
    AttemptedIntegerLiteral {
        #[primary_span]
        start_span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("invalid update of the `#[loop_match]` state")]
pub(crate) struct LoopMatchInvalidUpdate {
    #[primary_span]
    pub lhs: Span,
    #[label("the assignment must update this variable")]
    pub scrutinee: Span,
}

#[derive(Diagnostic)]
#[diag("invalid match on `#[loop_match]` state")]
#[note("a local variable must be the scrutinee within a `#[loop_match]`")]
pub(crate) struct LoopMatchInvalidMatch {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("this `#[loop_match]` state value has type `{$ty}`, which is not supported")]
#[note("only integers, floats, bool, char, and enums without fields are supported")]
pub(crate) struct LoopMatchUnsupportedType<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("statements are not allowed in this position within a `#[loop_match]`")]
pub(crate) struct LoopMatchBadStatements {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("this expression must be a single `match` wrapped in a labeled block")]
pub(crate) struct LoopMatchBadRhs {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected a single assignment expression")]
pub(crate) struct LoopMatchMissingAssignment {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("match arms that are part of a `#[loop_match]` cannot have guards")]
pub(crate) struct LoopMatchArmWithGuard {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("could not determine the target branch for this `#[const_continue]`")]
#[help("try extracting the expression into a `const` item")]
pub(crate) struct ConstContinueNotMonomorphicConst {
    #[primary_span]
    pub span: Span,

    #[subdiagnostic]
    pub reason: ConstContinueNotMonomorphicConstReason,
}

#[derive(Subdiagnostic)]
pub(crate) enum ConstContinueNotMonomorphicConstReason {
    #[label("constant parameters may use generics, and are not evaluated early enough")]
    ConstantParameter {
        #[primary_span]
        span: Span,
    },

    #[label("`const` blocks may use generics, and are not evaluated early enough")]
    ConstBlock {
        #[primary_span]
        span: Span,
    },

    #[label("this value must be a literal or a monomorphic const")]
    Other {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("could not determine the target branch for this `#[const_continue]`")]
pub(crate) struct ConstContinueBadConst {
    #[primary_span]
    #[label("this value is too generic")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("a `#[const_continue]` must break to a label with a value")]
pub(crate) struct ConstContinueMissingLabelOrValue {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("the target of this `#[const_continue]` is not statically known")]
pub(crate) struct ConstContinueUnknownJumpTarget {
    #[primary_span]
    pub span: Span,
}
