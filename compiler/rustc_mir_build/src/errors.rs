use crate::thir::pattern::MatchCheckCtxt;
use rustc_errors::{error_code, Applicability, DiagnosticBuilder, ErrorGuaranteed, MultiSpan};
use rustc_macros::{LintDiagnostic, SessionDiagnostic, SessionSubdiagnostic};
use rustc_middle::ty::{self, Ty};
use rustc_session::{parse::ParseSess, SessionDiagnostic};
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

#[derive(SessionDiagnostic)]
#[diag(mir_build::call_to_unsafe_fn_requires_unsafe, code = "E0133")]
#[note]
pub struct CallToUnsafeFunctionRequiresUnsafe<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub function: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::call_to_unsafe_fn_requires_unsafe_nameless, code = "E0133")]
#[note]
pub struct CallToUnsafeFunctionRequiresUnsafeNameless {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::call_to_unsafe_fn_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct CallToUnsafeFunctionRequiresUnsafeUnsafeOpInUnsafeFnAllowed<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub function: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(
    mir_build::call_to_unsafe_fn_requires_unsafe_nameless_unsafe_op_in_unsafe_fn_allowed,
    code = "E0133"
)]
#[note]
pub struct CallToUnsafeFunctionRequiresUnsafeNamelessUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::inline_assembly_requires_unsafe, code = "E0133")]
#[note]
pub struct UseOfInlineAssemblyRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::inline_assembly_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct UseOfInlineAssemblyRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::initializing_type_with_requires_unsafe, code = "E0133")]
#[note]
pub struct InitializingTypeWithRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(
    mir_build::initializing_type_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,
    code = "E0133"
)]
#[note]
pub struct InitializingTypeWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::mutable_static_requires_unsafe, code = "E0133")]
#[note]
pub struct UseOfMutableStaticRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::mutable_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct UseOfMutableStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::extern_static_requires_unsafe, code = "E0133")]
#[note]
pub struct UseOfExternStaticRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::extern_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct UseOfExternStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::deref_raw_pointer_requires_unsafe, code = "E0133")]
#[note]
pub struct DerefOfRawPointerRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::deref_raw_pointer_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct DerefOfRawPointerRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::union_field_requires_unsafe, code = "E0133")]
#[note]
pub struct AccessToUnionFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::union_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct AccessToUnionFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::mutation_of_layout_constrained_field_requires_unsafe, code = "E0133")]
#[note]
pub struct MutationOfLayoutConstrainedFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(
    mir_build::mutation_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,
    code = "E0133"
)]
#[note]
pub struct MutationOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::borrow_of_layout_constrained_field_requires_unsafe, code = "E0133")]
#[note]
pub struct BorrowOfLayoutConstrainedFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(
    mir_build::borrow_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,
    code = "E0133"
)]
#[note]
pub struct BorrowOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::call_to_fn_with_requires_unsafe, code = "E0133")]
#[note]
pub struct CallToFunctionWithRequiresUnsafe<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub function: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::call_to_fn_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct CallToFunctionWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub function: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unused_unsafe)]
pub struct UnusedUnsafe {
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub enclosing: Option<UnusedUnsafeEnclosing>,
}

#[derive(SessionSubdiagnostic)]
pub enum UnusedUnsafeEnclosing {
    #[label(mir_build::unused_unsafe_enclosing_block_label)]
    Block {
        #[primary_span]
        span: Span,
    },
    #[label(mir_build::unused_unsafe_enclosing_fn_label)]
    Function {
        #[primary_span]
        span: Span,
    },
}

pub(crate) struct NonExhaustivePatternsTypeNotEmpty<'p, 'tcx, 'm> {
    pub cx: &'m MatchCheckCtxt<'p, 'tcx>,
    pub expr_span: Span,
    pub span: Span,
    pub ty: Ty<'tcx>,
}

impl<'a> SessionDiagnostic<'a> for NonExhaustivePatternsTypeNotEmpty<'_, '_, '_> {
    fn into_diagnostic(self, sess: &'a ParseSess) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = sess.span_diagnostic.struct_span_err_with_code(
            self.span,
            rustc_errors::fluent::mir_build::non_exhaustive_patterns_type_not_empty,
            error_code!(E0004),
        );

        let peeled_ty = self.ty.peel_refs();
        diag.set_arg("ty", self.ty);
        diag.set_arg("peeled_ty", peeled_ty);

        if let ty::Adt(def, _) = peeled_ty.kind() {
            let def_span = self
                .cx
                .tcx
                .hir()
                .get_if_local(def.did())
                .and_then(|node| node.ident())
                .map(|ident| ident.span)
                .unwrap_or_else(|| self.cx.tcx.def_span(def.did()));

            // workaround to make test pass
            let mut span: MultiSpan = def_span.into();
            span.push_span_label(def_span, "");

            diag.span_note(span, rustc_errors::fluent::mir_build::def_note);
        }

        let is_variant_list_non_exhaustive = match self.ty.kind() {
            ty::Adt(def, _) if def.is_variant_list_non_exhaustive() && !def.did().is_local() => {
                true
            }
            _ => false,
        };

        if is_variant_list_non_exhaustive {
            diag.note(rustc_errors::fluent::mir_build::non_exhaustive_type_note);
        } else {
            diag.note(rustc_errors::fluent::mir_build::type_note);
        }

        if let ty::Ref(_, sub_ty, _) = self.ty.kind() {
            if self.cx.tcx.is_ty_uninhabited_from(self.cx.module, *sub_ty, self.cx.param_env) {
                diag.note(rustc_errors::fluent::mir_build::reference_note);
            }
        }

        let mut suggestion = None;
        let sm = self.cx.tcx.sess.source_map();
        if self.span.eq_ctxt(self.expr_span) {
            // Get the span for the empty match body `{}`.
            let (indentation, more) = if let Some(snippet) = sm.indentation_before(self.span) {
                (format!("\n{}", snippet), "    ")
            } else {
                (" ".to_string(), "")
            };
            suggestion = Some((
                self.span.shrink_to_hi().with_hi(self.expr_span.hi()),
                format!(
                    " {{{indentation}{more}_ => todo!(),{indentation}}}",
                    indentation = indentation,
                    more = more,
                ),
            ));
        }

        if let Some((span, sugg)) = suggestion {
            diag.span_suggestion_verbose(
                span,
                rustc_errors::fluent::mir_build::suggestion,
                sugg,
                Applicability::HasPlaceholders,
            );
        } else {
            diag.help(rustc_errors::fluent::mir_build::help);
        }

        diag
    }
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::static_in_pattern, code = "E0158")]
pub struct StaticInPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::assoc_const_in_pattern, code = "E0158")]
pub struct AssocConstInPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::const_param_in_pattern, code = "E0158")]
pub struct ConstParamInPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_build::non_const_path, code = "E0080")]
pub struct NonConstPath {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build::unreachable_pattern)]
pub struct UnreachablePattern {
    #[label]
    pub span: Option<Span>,
    #[label(mir_build::catchall_label)]
    pub catchall: Option<Span>,
}
