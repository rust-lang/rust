use rustc_errors::{fluent, ErrorGuaranteed, Handler};
use rustc_macros::SessionDiagnostic;
use rustc_middle::ty::{PolyTraitRef, Ty, Unevaluated};
use rustc_session::{Limit, SessionDiagnostic};
use rustc_span::{Span, Symbol};

#[derive(SessionDiagnostic)]
#[diag(trait_selection::dump_vtable_entries)]
pub struct DumpVTableEntries<'a> {
    #[primary_span]
    pub span: Span,
    pub trait_ref: PolyTraitRef<'a>,
    pub entries: String,
}

#[derive(SessionDiagnostic)]
#[diag(trait_selection::unable_to_construct_constant_value)]
pub struct UnableToConstructConstantValue<'a> {
    #[primary_span]
    pub span: Span,
    pub unevaluated: Unevaluated<'a>,
}

#[derive(SessionDiagnostic)]
#[help]
#[diag(trait_selection::auto_deref_reached_recursion_limit, code = "E0055")]
pub struct AutoDerefReachedRecursionLimit<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub ty: Ty<'a>,
    pub suggested_limit: Limit,
    pub crate_name: Symbol,
}

#[derive(SessionDiagnostic)]
#[diag(trait_selection::empty_on_clause_in_rustc_on_unimplemented, code = "E0232")]
pub struct EmptyOnClauseInOnUnimplemented {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(trait_selection::invalid_on_clause_in_rustc_on_unimplemented, code = "E0232")]
pub struct InvalidOnClauseInOnUnimplemented {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(trait_selection::no_value_in_rustc_on_unimplemented, code = "E0232")]
#[note]
pub struct NoValueInOnUnimplemented {
    #[primary_span]
    #[label]
    pub span: Span,
}

pub struct NegativePositiveConflict<'a> {
    pub impl_span: Span,
    pub trait_desc: &'a str,
    pub self_desc: &'a Option<String>,
    pub negative_impl_span: Result<Span, Symbol>,
    pub positive_impl_span: Result<Span, Symbol>,
}

impl SessionDiagnostic<'_> for NegativePositiveConflict<'_> {
    fn into_diagnostic(
        self,
        handler: &Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_err(fluent::trait_selection::negative_positive_conflict);
        diag.set_arg("trait_desc", self.trait_desc);
        diag.set_arg(
            "self_desc",
            self.self_desc.clone().map_or_else(|| String::from("none"), |ty| ty),
        );
        diag.set_span(self.impl_span);
        diag.code(rustc_errors::error_code!(E0751));
        match self.negative_impl_span {
            Ok(span) => {
                diag.span_label(span, fluent::trait_selection::negative_implementation_here);
            }
            Err(cname) => {
                diag.note(fluent::trait_selection::negative_implementation_in_crate);
                diag.set_arg("negative_impl_cname", cname.to_string());
            }
        }
        match self.positive_impl_span {
            Ok(span) => {
                diag.span_label(span, fluent::trait_selection::positive_implementation_here);
            }
            Err(cname) => {
                diag.note(fluent::trait_selection::positive_implementation_in_crate);
                diag.set_arg("positive_impl_cname", cname.to_string());
            }
        }
        diag
    }
}
