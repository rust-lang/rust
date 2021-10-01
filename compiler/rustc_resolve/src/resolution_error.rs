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
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0401,
            "can't use generic parameters from outer function",
        );
        err.span_label(self.span, "use of generic parameter from outer function".to_string());

        let sm = self.resolver.session.source_map();
        match outer_res {
            Res::SelfTy(maybe_trait_defid, maybe_impl_defid) => {
                if let Some(impl_span) =
                    maybe_impl_defid.and_then(|(def_id, _)| self.resolver.opt_span(def_id))
                {
                    err.span_label(
                        reduce_impl_span_to_impl_keyword(sm, impl_span),
                        "`Self` type implicitly declared here, by this `impl`",
                    );
                }
                match (maybe_trait_defid, maybe_impl_defid) {
                    (Some(_), None) => {
                        err.span_label(self.span, "can't use `Self` here");
                    }
                    (_, Some(_)) => {
                        err.span_label(self.span, "use a type here instead");
                    }
                    (None, None) => bug!("`impl` without trait nor type?"),
                }
                return err;
            }
            Res::Def(DefKind::TyParam, def_id) => {
                if let Some(span) = self.resolver.opt_span(def_id) {
                    err.span_label(span, "type parameter from outer function");
                }
            }
            Res::Def(DefKind::ConstParam, def_id) => {
                if let Some(span) = self.resolver.opt_span(def_id) {
                    err.span_label(span, "const parameter from outer function");
                }
            }
            _ => {
                bug!(
                    "GenericParamsFromOuterFunction should only be used with Res::SelfTy, \
                    DefKind::TyParam or DefKind::ConstParam"
                );
            }
        }

        if has_generic_params == HasGenericParams::Yes {
            // Try to retrieve the span of the function signature and generate a new
            // message with a local type or const parameter.
            let sugg_msg = "try using a local generic parameter instead";
            if let Some((sugg_span, snippet)) = sm.generate_local_type_param_snippet(self.span) {
                // Suggest the modification to the user
                err.span_suggestion(sugg_span, sugg_msg, snippet, Applicability::MachineApplicable);
            } else if let Some(sp) = sm.generate_fn_name_span(self.span) {
                err.span_label(
                    sp,
                    "try adding a local generic parameter in this method instead".to_string(),
                );
            } else {
                err.help("try using a local generic parameter instead");
            }
        }

        err
    }

    fn name_already_used_in_parameter_list(
        &self,
        name: Symbol,
        first_use_span: Span,
    ) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0403,
            "the name `{}` is already used for a generic \
                parameter in this item's generic parameters",
            name,
        );
        err.span_label(self.span, "already used");
        err.span_label(first_use_span, format!("first use of `{}`", name));
        err
    }

    fn method_not_member_of_trait(
        &self,
        method: Ident,
        trait_: &str,
        candidate: Option<Symbol>,
    ) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0407,
            "method `{}` is not a member of trait `{}`",
            method,
            trait_
        );
        err.span_label(self.span, format!("not a member of trait `{}`", trait_));
        if let Some(candidate) = candidate {
            err.span_suggestion(
                method.span,
                "there is an associated function with a similar name",
                candidate.to_ident_string(),
                Applicability::MaybeIncorrect,
            );
        }
        err
    }

    fn type_not_member_of_trait(
        &self,
        type_: Ident,
        trait_: &str,
        candidate: Option<Symbol>,
    ) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0437,
            "type `{}` is not a member of trait `{}`",
            type_,
            trait_
        );
        err.span_label(self.span, format!("not a member of trait `{}`", trait_));
        if let Some(candidate) = candidate {
            err.span_suggestion(
                type_.span,
                "there is an associated type with a similar name",
                candidate.to_ident_string(),
                Applicability::MaybeIncorrect,
            );
        }
        err
    }

    fn const_not_member_of_trait(
        &self,
        const_: Ident,
        trait_: &str,
        candidate: Option<Symbol>,
    ) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0438,
            "const `{}` is not a member of trait `{}`",
            const_,
            trait_
        );
        err.span_label(self.span, format!("not a member of trait `{}`", trait_));
        if let Some(candidate) = candidate {
            err.span_suggestion(
                const_.span,
                "there is an associated constant with a similar name",
                candidate.to_ident_string(),
                Applicability::MaybeIncorrect,
            );
        }
        err
    }

    fn variable_not_bound_in_pattern(&self, binding_error: &BindingError) -> DiagnosticBuilder<'b> {
        let BindingError { name, target, origin, could_be_path } = binding_error;

        let target_sp = target.iter().copied().collect::<Vec<_>>();
        let origin_sp = origin.iter().copied().collect::<Vec<_>>();

        let msp = MultiSpan::from_spans(target_sp.clone());
        let mut err = struct_span_err!(
            self.resolver.session,
            msp,
            E0408,
            "variable `{}` is not bound in all patterns",
            name,
        );
        for sp in target_sp {
            err.span_label(sp, format!("pattern doesn't bind `{}`", name));
        }
        for sp in origin_sp {
            err.span_label(sp, "variable not in all patterns");
        }
        if *could_be_path {
            let help_msg = format!(
                "if you meant to match on a variant or a `const` item, consider \
                    making the path in the pattern qualified: `?::{}`",
                name,
            );
            err.span_help(self.span, &help_msg);
        }
        err
    }

    fn variable_bound_with_different_mode(
        &self,
        variable_name: Symbol,
        first_binding_span: Span,
    ) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0409,
            "variable `{}` is bound inconsistently across alternatives separated by `|`",
            variable_name
        );
        err.span_label(self.span, "bound in different ways");
        err.span_label(first_binding_span, "first binding");
        err
    }

    fn identifier_bound_more_than_once_in_parameter_list(
        &self,
        identifier: Symbol,
    ) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0415,
            "identifier `{}` is bound more than once in this parameter list",
            identifier
        );
        err.span_label(self.span, "used as parameter more than once");
        err
    }

    fn identifier_bound_more_than_once_in_same_pattern(
        &self,
        identifier: Symbol,
    ) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0416,
            "identifier `{}` is bound more than once in the same pattern",
            identifier
        );
        err.span_label(self.span, "used in a pattern more than once");
        err
    }

    fn undeclared_label(
        &self,
        name: Symbol,
        suggestion: Option<LabelSuggestion>,
    ) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0426,
            "use of undeclared label `{}`",
            name
        );

        err.span_label(self.span, format!("undeclared label `{}`", name));

        match suggestion {
            // A reachable label with a similar name exists.
            Some((ident, true)) => {
                err.span_label(ident.span, "a label with a similar name is reachable");
                err.span_suggestion(
                    self.span,
                    "try using similarly named label",
                    ident.name.to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
            // An unreachable label with a similar name exists.
            Some((ident, false)) => {
                err.span_label(ident.span, "a label with a similar name exists but is unreachable");
            }
            // No similarly-named labels exist.
            None => (),
        }

        err
    }

    fn self_imports_only_allowed_within(
        &self,
        root: bool,
        span_with_rename: Span,
    ) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0429,
            "{}",
            "`self` imports are only allowed within a { } list"
        );

        // None of the suggestions below would help with a case like `use self`.
        if !root {
            // use foo::bar::self        -> foo::bar
            // use foo::bar::self as abc -> foo::bar as abc
            err.span_suggestion(
                self.span,
                "consider importing the module directly",
                "".to_string(),
                Applicability::MachineApplicable,
            );

            // use foo::bar::self        -> foo::bar::{self}
            // use foo::bar::self as abc -> foo::bar::{self as abc}
            let braces = vec![
                (span_with_rename.shrink_to_lo(), "{".to_string()),
                (span_with_rename.shrink_to_hi(), "}".to_string()),
            ];
            err.multipart_suggestion(
                "alternatively, use the multi-path `use` syntax to import `self`",
                braces,
                Applicability::MachineApplicable,
            );
        }
        err
    }

    fn self_import_can_only_appear_once_in_the_list(&self) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0430,
            "`self` import can only appear once in an import list"
        );
        err.span_label(self.span, "can only appear once in an import list");
        err
    }

    fn self_import_only_in_import_list_with_non_empty_prefix(&self) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0431,
            "`self` import can only appear in an import list with \
                                                a non-empty prefix"
        );
        err.span_label(self.span, "can only appear in an import list with a non-empty prefix");
        err
    }

    fn failed_to_resolve(
        &self,
        label: String,
        suggestion: Option<Suggestion>,
    ) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0433,
            "failed to resolve: {}",
            &label
        );
        err.span_label(self.span, label);

        if let Some((suggestions, msg, applicability)) = suggestion {
            err.multipart_suggestion(&msg, suggestions, applicability);
        }

        err
    }

    fn cannot_capture_dynamic_environment_in_fn_item(&self) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0434,
            "{}",
            "can't capture dynamic environment in a fn item"
        );
        err.help("use the `|| { ... }` closure form instead");
        err
    }

    fn attempt_to_use_non_constant_value_in_constant(
        &self,
        ident: Ident,
        sugg: &str,
        current: &str,
    ) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0435,
            "attempt to use a non-constant value in a constant"
        );
        // let foo =...
        //     ^^^ given this Span
        // ------- get this Span to have an applicable suggestion
        let sp =
            self.resolver.session.source_map().span_extend_to_prev_str(ident.span, current, true);
        if sp.lo().0 == 0 {
            err.span_label(ident.span, &format!("this would need to be a `{}`", sugg));
        } else {
            let sp = sp.with_lo(BytePos(sp.lo().0 - current.len() as u32));
            err.span_suggestion(
                sp,
                &format!("consider using `{}` instead of `{}`", sugg, current),
                format!("{} {}", sugg, ident),
                Applicability::MaybeIncorrect,
            );
            err.span_label(self.span, "non-constant value");
        }
        err
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
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0530,
            "{}s cannot shadow {}s",
            shadowing_binding_descr,
            shadowed_binding_descr,
        );
        err.span_label(
            self.span,
            format!("cannot be named the same as {} {}", article, shadowed_binding_descr),
        );
        let msg = format!("the {} `{}` is {} here", shadowed_binding_descr, name, participle); /* test */
        err.span_label(shadowed_binding_span, msg);
        err
    }

    fn forward_declared_generic_param(&self) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0128,
            "generic parameters with a default cannot use \
                                                forward declared identifiers"
        );
        err.span_label(
            self.span,
            "defaulted generic parameters cannot be forward declared".to_string(),
        );
        err
    }

    fn param_in_ty_of_const_param(&self, name: Symbol) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0770,
            "the type of const parameters must not depend on other generic parameters"
        );
        err.span_label(self.span, format!("the type must not depend on the parameter `{}`", name));
        err
    }

    fn param_in_non_trivial_anon_const(
        &self,
        name: Symbol,
        is_type: bool,
    ) -> DiagnosticBuilder<'b> {
        let mut err = self
            .resolver
            .session
            .struct_span_err(self.span, "generic parameters may not be used in const operations");
        err.span_label(self.span, &format!("cannot perform const operation using `{}`", name));

        if is_type {
            err.note("type parameters may not be used in const expressions");
        } else {
            err.help(&format!(
                "const parameters may only be used as standalone arguments, i.e. `{}`",
                name
            ));
        }

        if self.resolver.session.is_nightly_build() {
            err.help("use `#![feature(generic_const_exprs)]` to allow generic const expressions");
        }

        err
    }

    fn self_in_generic_param_default(&self) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0735,
            "generic parameters cannot use `Self` in their defaults"
        );
        err.span_label(self.span, "`Self` in generic parameter default".to_string());
        err
    }

    fn unreachable_label(
        &self,
        name: Symbol,
        definition_span: Span,
        suggestion: Option<LabelSuggestion>,
    ) -> DiagnosticBuilder<'b> {
        let mut err = struct_span_err!(
            self.resolver.session,
            self.span,
            E0767,
            "use of unreachable label `{}`",
            name,
        );

        err.span_label(definition_span, "unreachable label defined here");
        err.span_label(self.span, format!("unreachable label `{}`", name));
        err.note("labels are unreachable through functions, closures, async blocks and modules");

        match suggestion {
            // A reachable label with a similar name exists.
            Some((ident, true)) => {
                err.span_label(ident.span, "a label with a similar name is reachable");
                err.span_suggestion(
                    self.span,
                    "try using similarly named label",
                    ident.name.to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
            // An unreachable label with a similar name exists.
            Some((ident, false)) => {
                err.span_label(
                    ident.span,
                    "a label with a similar name exists but is also unreachable",
                );
            }
            // No similarly-named labels exist.
            None => (),
        }

        err
    }
}
