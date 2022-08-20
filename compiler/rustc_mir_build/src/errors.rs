use rustc_macros::LintDiagnostic;
use rustc_span::Span;

#[derive(LintDiagnostic)]
#[diag(mir_build::unconditional_recursion)]
#[help]
pub struct UnconditionalRecursion {
    #[label]
    pub span: Span,
    #[label(mir_build::unconditional_recursion_call_site_label)]
    pub call_sites: Vec<Span>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafe<'a> {
    #[label]
    pub span: Span,
    pub function: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe_nameless)]
#[note]
pub struct UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafeNameless {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unsafe_op_in_unsafe_fn_inline_assembly_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnUseOfInlineAssemblyRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unsafe_op_in_unsafe_fn_initializing_type_with_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnInitializingTypeWithRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unsafe_op_in_unsafe_fn_mutable_static_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnUseOfMutableStaticRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unsafe_op_in_unsafe_fn_extern_static_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnUseOfExternStaticRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unsafe_op_in_unsafe_fn_deref_raw_pointer_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnDerefOfRawPointerRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unsafe_op_in_unsafe_fn_union_field_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnAccessToUnionFieldRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unsafe_op_in_unsafe_fn_mutation_of_layout_constrained_field_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnMutationOfLayoutConstrainedFieldRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unsafe_op_in_unsafe_fn_borrow_of_layout_constrained_field_requires_unsafe)]
pub struct UnsafeOpInUnsafeFnBorrowOfLayoutConstrainedFieldRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unsafe_op_in_unsafe_fn_call_to_fn_with_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnCallToFunctionWithRequiresUnsafe<'a> {
    #[label]
    pub span: Span,
    pub function: &'a str,
}
