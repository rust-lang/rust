use rustc_ast::TraitObjectSyntax;
use rustc_errors::{Diagnostic, StashKey};
use rustc_hir as hir;
use rustc_lint_defs::{builtin::BARE_TRAIT_OBJECTS, Applicability};
use rustc_span::Span;
use rustc_trait_selection::traits::error_reporting::suggestions::NextTypeParamName;

use super::AstConv;

impl<'o, 'tcx> dyn AstConv<'tcx> + 'o {
    /// Make sure that we are in the condition to suggest the blanket implementation.
    pub(super) fn maybe_lint_blanket_trait_impl(
        &self,
        self_ty: &hir::Ty<'_>,
        diag: &mut Diagnostic,
    ) {
        let tcx = self.tcx();
        let parent_id = tcx.hir().get_parent_item(self_ty.hir_id).def_id;
        if let hir::Node::Item(hir::Item {
            kind:
                hir::ItemKind::Impl(hir::Impl {
                    self_ty: impl_self_ty,
                    of_trait: Some(of_trait_ref),
                    generics,
                    ..
                }),
            ..
        }) = tcx.hir_node_by_def_id(parent_id)
            && self_ty.hir_id == impl_self_ty.hir_id
        {
            if !of_trait_ref.trait_def_id().is_some_and(|def_id| def_id.is_local()) {
                return;
            }
            let of_trait_span = of_trait_ref.path.span;
            // make sure that we are not calling unwrap to abort during the compilation
            let Ok(of_trait_name) = tcx.sess.source_map().span_to_snippet(of_trait_span) else {
                return;
            };

            let Ok(impl_trait_name) = self.tcx().sess.source_map().span_to_snippet(self_ty.span)
            else {
                return;
            };
            let Some(sugg) = self.generics_suggestion(generics, self_ty.span, &impl_trait_name)
            else {
                return;
            };
            diag.multipart_suggestion(
                format!(
                    "alternatively use a blanket implementation to implement `{of_trait_name}` for \
                     all types that also implement `{impl_trait_name}`"
                ),
                sugg,
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn generics_suggestion(
        &self,
        generics: &hir::Generics<'_>,
        self_ty_span: Span,
        impl_trait_name: &str,
    ) -> Option<Vec<(Span, String)>> {
        // check if the trait has generics, to make a correct suggestion
        let param_name = generics.params.next_type_param_name(None);

        let add_generic_sugg = if let Some(span) = generics.span_for_param_suggestion() {
            (span, format!(", {param_name}: {impl_trait_name}"))
        } else {
            (generics.span, format!("<{param_name}: {impl_trait_name}>"))
        };
        Some(vec![(self_ty_span, param_name), add_generic_sugg])
    }

    /// Make sure that we are in the condition to suggest `impl Trait`.
    fn maybe_lint_impl_trait(&self, self_ty: &hir::Ty<'_>, diag: &mut Diagnostic) -> bool {
        let tcx = self.tcx();
        let parent_id = tcx.hir().get_parent_item(self_ty.hir_id).def_id;
        let (hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(sig, generics, _), .. })
        | hir::Node::TraitItem(hir::TraitItem {
            kind: hir::TraitItemKind::Fn(sig, _),
            generics,
            ..
        })) = tcx.hir_node_by_def_id(parent_id)
        else {
            return false;
        };
        let Ok(trait_name) = self.tcx().sess.source_map().span_to_snippet(self_ty.span) else {
            return false;
        };
        let impl_sugg = vec![(self_ty.span.shrink_to_lo(), "impl ".to_string())];
        if let hir::FnRetTy::Return(ty) = sig.decl.output
            && ty.hir_id == self_ty.hir_id
        {
            diag.multipart_suggestion_verbose(
                format!("use `impl {trait_name}` to return an opaque type, as long as you return a single underlying type"),
                impl_sugg,
                Applicability::MachineApplicable,
            );
            diag.multipart_suggestion_verbose(
                "alternatively, you can return an owned trait object",
                vec![
                    (ty.span.shrink_to_lo(), "Box<dyn ".to_string()),
                    (ty.span.shrink_to_hi(), ">".to_string()),
                ],
                Applicability::MachineApplicable,
            );
            return true;
        }
        for ty in sig.decl.inputs {
            if ty.hir_id == self_ty.hir_id {
                if let Some(sugg) = self.generics_suggestion(generics, self_ty.span, &trait_name) {
                    diag.multipart_suggestion_verbose(
                        format!("use a new generic type parameter, constrained by `{trait_name}`"),
                        sugg,
                        Applicability::MachineApplicable,
                    );
                    diag.multipart_suggestion_verbose(
                        "you can also use an opaque type, but users won't be able to specify the \
                         type parameter when calling the `fn`, having to rely exclusively on type \
                         inference",
                        impl_sugg,
                        Applicability::MachineApplicable,
                    );
                }
                let sugg = if let hir::TyKind::TraitObject([_, _, ..], _, _) = self_ty.kind {
                    // There are more than one trait bound, we need surrounding parentheses.
                    vec![
                        (self_ty.span.shrink_to_lo(), "&(dyn ".to_string()),
                        (self_ty.span.shrink_to_hi(), ")".to_string()),
                    ]
                } else {
                    vec![(self_ty.span.shrink_to_lo(), "&dyn ".to_string())]
                };
                diag.multipart_suggestion_verbose(
                    format!(
                        "alternatively, use a trait object to accept any type that implements \
                         `{trait_name}`, accessing its methods at runtime using dynamic dispatch",
                    ),
                    sugg,
                    Applicability::MachineApplicable,
                );
                return true;
            }
        }
        false
    }

    pub(super) fn maybe_lint_bare_trait(&self, self_ty: &hir::Ty<'_>, in_path: bool) {
        let tcx = self.tcx();
        if let hir::TyKind::TraitObject([poly_trait_ref, ..], _, TraitObjectSyntax::None) =
            self_ty.kind
        {
            let needs_bracket = in_path
                && !tcx
                    .sess
                    .source_map()
                    .span_to_prev_source(self_ty.span)
                    .ok()
                    .is_some_and(|s| s.trim_end().ends_with('<'));

            let is_global = poly_trait_ref.trait_ref.path.is_global();

            let mut sugg = Vec::from_iter([(
                self_ty.span.shrink_to_lo(),
                format!(
                    "{}dyn {}",
                    if needs_bracket { "<" } else { "" },
                    if is_global { "(" } else { "" },
                ),
            )]);

            if is_global || needs_bracket {
                sugg.push((
                    self_ty.span.shrink_to_hi(),
                    format!(
                        "{}{}",
                        if is_global { ")" } else { "" },
                        if needs_bracket { ">" } else { "" },
                    ),
                ));
            }

            if self_ty.span.edition().at_least_rust_2021() {
                let msg = "trait objects must include the `dyn` keyword";
                let label = "add `dyn` keyword before this trait";
                let mut diag =
                    rustc_errors::struct_span_err!(tcx.dcx(), self_ty.span, E0782, "{}", msg);
                if self_ty.span.can_be_used_for_suggestions()
                    && !self.maybe_lint_impl_trait(self_ty, &mut diag)
                {
                    diag.multipart_suggestion_verbose(
                        label,
                        sugg,
                        Applicability::MachineApplicable,
                    );
                }
                // check if the impl trait that we are considering is a impl of a local trait
                self.maybe_lint_blanket_trait_impl(self_ty, &mut diag);
                diag.stash(self_ty.span, StashKey::TraitMissingMethod);
            } else {
                let msg = "trait objects without an explicit `dyn` are deprecated";
                tcx.struct_span_lint_hir(
                    BARE_TRAIT_OBJECTS,
                    self_ty.hir_id,
                    self_ty.span,
                    msg,
                    |lint| {
                        if self_ty.span.can_be_used_for_suggestions()
                            && !self.maybe_lint_impl_trait(self_ty, lint)
                        {
                            lint.multipart_suggestion_verbose(
                                "use `dyn`",
                                sugg,
                                Applicability::MachineApplicable,
                            );
                        }
                        self.maybe_lint_blanket_trait_impl(self_ty, lint);
                    },
                );
            }
        }
    }
}
