use rustc_ast::TraitObjectSyntax;
use rustc_errors::codes::*;
use rustc_errors::{Diag, EmissionGuarantee, ErrorGuaranteed, StashKey, Suggestions};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_lint_defs::Applicability;
use rustc_lint_defs::builtin::BARE_TRAIT_OBJECTS;
use rustc_span::Span;
use rustc_trait_selection::error_reporting::traits::suggestions::NextTypeParamName;

use super::HirTyLowerer;

impl<'tcx> dyn HirTyLowerer<'tcx> + '_ {
    /// Prohibit or lint against *bare* trait object types depending on the edition.
    ///
    /// *Bare* trait object types are ones that aren't preceded by the keyword `dyn`.
    /// In edition 2021 and onward we emit a hard error for them.
    pub(super) fn prohibit_or_lint_bare_trait_object_ty(
        &self,
        self_ty: &hir::Ty<'_>,
    ) -> Option<ErrorGuaranteed> {
        let tcx = self.tcx();

        let hir::TyKind::TraitObject([poly_trait_ref, ..], _, TraitObjectSyntax::None) =
            self_ty.kind
        else {
            return None;
        };

        let in_path = match tcx.parent_hir_node(self_ty.hir_id) {
            hir::Node::Ty(hir::Ty {
                kind: hir::TyKind::Path(hir::QPath::TypeRelative(qself, _)),
                ..
            })
            | hir::Node::Expr(hir::Expr {
                kind: hir::ExprKind::Path(hir::QPath::TypeRelative(qself, _)),
                ..
            })
            | hir::Node::Pat(hir::Pat {
                kind: hir::PatKind::Path(hir::QPath::TypeRelative(qself, _)),
                ..
            }) if qself.hir_id == self_ty.hir_id => true,
            _ => false,
        };
        let needs_bracket = in_path
            && !tcx
                .sess
                .source_map()
                .span_to_prev_source(self_ty.span)
                .ok()
                .is_some_and(|s| s.trim_end().ends_with('<'));

        let is_global = poly_trait_ref.trait_ref.path.is_global();

        let mut sugg = vec![(
            self_ty.span.shrink_to_lo(),
            format!(
                "{}dyn {}",
                if needs_bracket { "<" } else { "" },
                if is_global { "(" } else { "" },
            ),
        )];

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
            let msg = "expected a type, found a trait";
            let label = "you can add the `dyn` keyword if you want a trait object";
            let mut diag =
                rustc_errors::struct_span_code_err!(self.dcx(), self_ty.span, E0782, "{}", msg);
            if self_ty.span.can_be_used_for_suggestions()
                && !self.maybe_suggest_impl_trait(self_ty, &mut diag)
            {
                // FIXME: Only emit this suggestion if the trait is dyn-compatible.
                diag.multipart_suggestion_verbose(label, sugg, Applicability::MachineApplicable);
            }
            // Check if the impl trait that we are considering is an impl of a local trait.
            self.maybe_suggest_blanket_trait_impl(self_ty, &mut diag);
            self.maybe_suggest_assoc_ty_bound(self_ty, &mut diag);
            // In case there is an associate type with the same name
            // Add the suggestion to this error
            if let Some(mut sugg) =
                tcx.dcx().steal_non_err(self_ty.span, StashKey::AssociatedTypeSuggestion)
                && let Suggestions::Enabled(ref mut s1) = diag.suggestions
                && let Suggestions::Enabled(ref mut s2) = sugg.suggestions
            {
                s1.append(s2);
                sugg.cancel();
            }
            diag.stash(self_ty.span, StashKey::TraitMissingMethod)
        } else {
            tcx.node_span_lint(BARE_TRAIT_OBJECTS, self_ty.hir_id, self_ty.span, |lint| {
                lint.primary_message("trait objects without an explicit `dyn` are deprecated");
                if self_ty.span.can_be_used_for_suggestions() {
                    lint.multipart_suggestion_verbose(
                        "if this is a dyn-compatible trait, use `dyn`",
                        sugg,
                        Applicability::MachineApplicable,
                    );
                }
                self.maybe_suggest_blanket_trait_impl(self_ty, lint);
            });
            None
        }
    }

    /// Make sure that we are in the condition to suggest the blanket implementation.
    fn maybe_suggest_blanket_trait_impl<G: EmissionGuarantee>(
        &self,
        self_ty: &hir::Ty<'_>,
        diag: &mut Diag<'_, G>,
    ) {
        let tcx = self.tcx();
        let parent_id = tcx.hir().get_parent_item(self_ty.hir_id).def_id;
        if let hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Impl(hir::Impl { self_ty: impl_self_ty, of_trait, generics, .. }),
            ..
        }) = tcx.hir_node_by_def_id(parent_id)
            && self_ty.hir_id == impl_self_ty.hir_id
        {
            let Some(of_trait_ref) = of_trait else {
                diag.span_suggestion_verbose(
                    impl_self_ty.span.shrink_to_hi(),
                    "you might have intended to implement this trait for a given type",
                    format!(" for /* Type */"),
                    Applicability::HasPlaceholders,
                );
                return;
            };
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
            let sugg = self.add_generic_param_suggestion(generics, self_ty.span, &impl_trait_name);
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

    fn add_generic_param_suggestion(
        &self,
        generics: &hir::Generics<'_>,
        self_ty_span: Span,
        impl_trait_name: &str,
    ) -> Vec<(Span, String)> {
        // check if the trait has generics, to make a correct suggestion
        let param_name = generics.params.next_type_param_name(None);

        let add_generic_sugg = if let Some(span) = generics.span_for_param_suggestion() {
            (span, format!(", {param_name}: {impl_trait_name}"))
        } else {
            (generics.span, format!("<{param_name}: {impl_trait_name}>"))
        };
        vec![(self_ty_span, param_name), add_generic_sugg]
    }

    /// Make sure that we are in the condition to suggest `impl Trait`.
    fn maybe_suggest_impl_trait(&self, self_ty: &hir::Ty<'_>, diag: &mut Diag<'_>) -> bool {
        let tcx = self.tcx();
        let parent_id = tcx.hir().get_parent_item(self_ty.hir_id).def_id;
        // FIXME: If `type_alias_impl_trait` is enabled, also look for `Trait0<Ty = Trait1>`
        //        and suggest `Trait0<Ty = impl Trait1>`.
        // Functions are found in three different contexts.
        // 1. Independent functions
        // 2. Functions inside trait blocks
        // 3. Functions inside impl blocks
        let (sig, generics) = match tcx.hir_node_by_def_id(parent_id) {
            hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(sig, generics, _), .. }) => {
                (sig, generics)
            }
            hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(sig, _),
                generics,
                ..
            }) => (sig, generics),
            hir::Node::ImplItem(hir::ImplItem {
                kind: hir::ImplItemKind::Fn(sig, _),
                generics,
                ..
            }) => (sig, generics),
            _ => return false,
        };
        let Ok(trait_name) = tcx.sess.source_map().span_to_snippet(self_ty.span) else {
            return false;
        };
        let impl_sugg = vec![(self_ty.span.shrink_to_lo(), "impl ".to_string())];
        // Check if trait object is safe for suggesting dynamic dispatch.
        let is_dyn_compatible = match self_ty.kind {
            hir::TyKind::TraitObject(objects, ..) => {
                objects.iter().all(|o| match o.trait_ref.path.res {
                    Res::Def(DefKind::Trait, id) => tcx.is_dyn_compatible(id),
                    _ => false,
                })
            }
            _ => false,
        };

        let borrowed = matches!(
            tcx.parent_hir_node(self_ty.hir_id),
            hir::Node::Ty(hir::Ty { kind: hir::TyKind::Ref(..), .. })
        );

        // Suggestions for function return type.
        if let hir::FnRetTy::Return(ty) = sig.decl.output
            && ty.peel_refs().hir_id == self_ty.hir_id
        {
            let pre = if !is_dyn_compatible {
                format!("`{trait_name}` is dyn-incompatible, ")
            } else {
                String::new()
            };
            let msg = format!(
                "{pre}use `impl {trait_name}` to return an opaque type, as long as you return a \
                 single underlying type",
            );

            diag.multipart_suggestion_verbose(msg, impl_sugg, Applicability::MachineApplicable);

            // Suggest `Box<dyn Trait>` for return type
            if is_dyn_compatible {
                // If the return type is `&Trait`, we don't want
                // the ampersand to be displayed in the `Box<dyn Trait>`
                // suggestion.
                let suggestion = if borrowed {
                    vec![(ty.span, format!("Box<dyn {trait_name}>"))]
                } else {
                    vec![
                        (ty.span.shrink_to_lo(), "Box<dyn ".to_string()),
                        (ty.span.shrink_to_hi(), ">".to_string()),
                    ]
                };

                diag.multipart_suggestion_verbose(
                    "alternatively, you can return an owned trait object",
                    suggestion,
                    Applicability::MachineApplicable,
                );
            }
            return true;
        }

        // Suggestions for function parameters.
        for ty in sig.decl.inputs {
            if ty.peel_refs().hir_id != self_ty.hir_id {
                continue;
            }
            let sugg = self.add_generic_param_suggestion(generics, self_ty.span, &trait_name);
            diag.multipart_suggestion_verbose(
                format!("use a new generic type parameter, constrained by `{trait_name}`"),
                sugg,
                Applicability::MachineApplicable,
            );
            diag.multipart_suggestion_verbose(
                "you can also use an opaque type, but users won't be able to specify the type \
                 parameter when calling the `fn`, having to rely exclusively on type inference",
                impl_sugg,
                Applicability::MachineApplicable,
            );
            if !is_dyn_compatible {
                diag.note(format!("`{trait_name}` it is dyn-incompatible, so it can't be `dyn`"));
            } else {
                // No ampersand in suggestion if it's borrowed already
                let (dyn_str, paren_dyn_str) =
                    if borrowed { ("dyn ", "(dyn ") } else { ("&dyn ", "&(dyn ") };

                let sugg = if let hir::TyKind::TraitObject([_, _, ..], _, _) = self_ty.kind {
                    // There are more than one trait bound, we need surrounding parentheses.
                    vec![
                        (self_ty.span.shrink_to_lo(), paren_dyn_str.to_string()),
                        (self_ty.span.shrink_to_hi(), ")".to_string()),
                    ]
                } else {
                    vec![(self_ty.span.shrink_to_lo(), dyn_str.to_string())]
                };
                diag.multipart_suggestion_verbose(
                    format!(
                        "alternatively, use a trait object to accept any type that implements \
                         `{trait_name}`, accessing its methods at runtime using dynamic dispatch",
                    ),
                    sugg,
                    Applicability::MachineApplicable,
                );
            }
            return true;
        }
        false
    }

    fn maybe_suggest_assoc_ty_bound(&self, self_ty: &hir::Ty<'_>, diag: &mut Diag<'_>) {
        let mut parents = self.tcx().hir().parent_iter(self_ty.hir_id);

        if let Some((_, hir::Node::AssocItemConstraint(constraint))) = parents.next()
            && let Some(obj_ty) = constraint.ty()
        {
            if let Some((_, hir::Node::TraitRef(..))) = parents.next()
                && let Some((_, hir::Node::Ty(ty))) = parents.next()
                && let hir::TyKind::TraitObject(..) = ty.kind
            {
                // Assoc ty bounds aren't permitted inside trait object types.
                return;
            }

            let lo = if constraint.gen_args.span_ext.is_dummy() {
                constraint.ident.span
            } else {
                constraint.gen_args.span_ext
            };
            let hi = obj_ty.span;

            if !lo.eq_ctxt(hi) {
                return;
            }

            diag.span_suggestion_verbose(
                lo.between(hi),
                "you might have meant to write a bound here",
                ": ",
                Applicability::MaybeIncorrect,
            );
        }
    }
}
