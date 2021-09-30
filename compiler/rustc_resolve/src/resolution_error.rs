use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir::def::DefKind;
use rustc_middle::bug;
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::{BytePos, MultiSpan, Span};

use crate::{BindingError, HasGenericParams};
use crate::{ResolutionError, Resolver};

use crate::diagnostics::{reduce_impl_span_to_impl_keyword, LabelSuggestion, Res, Suggestion};

struct StructErrorBuilder<'a, 'b> {
    resolver: &'a Resolver<'b>,
    span: Span,
}

impl<'a, 'b> StructErrorBuilder<'a, 'b> {
    fn generic_params_from_outer_function(
        &self,
        outer_res: Res,
        has_generic_params: HasGenericParams,
    ) -> DiagnosticBuilder<'b> {
    }

    fn name_already_used_in_parameter_list(
        &self,
        name: Symbol,
        first_use_span: Span,
    ) -> DiagnosticBuilder<'b> {
    }

    fn method_not_member_of_trait(
        &self,
        method: Ident,
        trait_: &str,
        candidate: Option<Symbol>,
    ) -> DiagnosticBuilder<'b> {
    }

    fn type_not_member_of_trait(
        &self,
        type_: Ident,
        trait_: &str,
        candidate: Option<Symbol>,
    ) -> DiagnosticBuilder<'b> {
    }

    fn const_not_member_of_trait(
        &self,
        const_: Ident,
        trait_: &str,
        candidate: Option<Symbol>,
    ) -> DiagnosticBuilder<'b> {
    }

    fn variable_not_bound_in_pattern(&self, binding_error: &BindingError) -> DiagnosticBuilder<'b> {
    }

    fn variable_bound_with_different_mode(
        &self,
        variable_name: Symbol,
        first_binding_span: Span,
    ) -> DiagnosticBuilder<'b> {
    }

    fn identifier_bound_more_than_once_in_parameter_list(
        &self,
        identifier: Symbol,
    ) -> DiagnosticBuilder<'b> {
    }

    fn identifier_bound_more_than_once_in_same_pattern(
        &self,
        identifier: Symbol,
    ) -> DiagnosticBuilder<'b> {
    }

    fn undeclared_label(
        &self,
        name: Symbol,
        suggestion: Option<LabelSuggestion>,
    ) -> DiagnosticBuilder<'b> {
    }

    fn self_imports_only_allowed_within(
        &self,
        root: bool,
        span_with_rename: Span,
    ) -> DiagnosticBuilder<'b> {
    }

    fn self_import_can_only_appear_once_in_the_list(&self) -> DiagnosticBuilder<'b> {}

    fn self_import_only_in_import_list_with_non_empty_prefix(&self) -> DiagnosticBuilder<'b> {}

    fn failed_to_resolve(
        &self,
        label: String,
        suggestion: Option<Suggestion>,
    ) -> DiagnosticBuilder<'b> {
    }

    fn cannot_capture_dynamic_environment_in_fn_item(&self) -> DiagnosticBuilder<'b> {}

    fn attempt_to_use_non_constant_value_in_constant(
        &self,
        ident: Ident,
        sugg: &str,
        current: &str,
    ) -> DiagnosticBuilder<'b> {
    }

    fn binding_shadows_something_unacceptable(
        &self,
        shadowing_binding_descr: &'static str,
        name: Symbol,
        participle: &'static str,
        article: &'static str,
        shadowed_binding_descr: &'static str,
        shadowed_binding_span: Span,
    ) -> DiagnosticBuilder<'b> {
    }

    fn forward_declared_generic_param(&self) -> DiagnosticBuilder<'b> {}

    fn param_in_ty_of_const_param(&self, name: Symbol) -> DiagnosticBuilder<'b> {}

    fn param_in_non_trivial_anon_const(
        &self,
        name: Symbol,
        is_type: bool,
    ) -> DiagnosticBuilder<'b> {
    }

    fn self_in_generic_param_default(&self) -> DiagnosticBuilder<'b> {}

    fn unreachable_label(
        &self,
        name: Symbol,
        definition_span: Span,
        suggestion: Option<LabelSuggestion>,
    ) -> DiagnosticBuilder<'b> {
    }
}
