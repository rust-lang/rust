use std::fmt::Write;

use hir::def_id::DefId;
use hir::{HirId, ItemKind};
use rustc_ast::join_path_idents;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{ARRAY_INTO_ITER, BOXED_SLICE_INTO_ITER};
use rustc_middle::span_bug;
use rustc_middle::ty::{self, Ty};
use rustc_session::lint::builtin::{RUST_2021_PRELUDE_COLLISIONS, RUST_2024_PRELUDE_COLLISIONS};
use rustc_span::{Ident, STDLIB_STABLE_CRATES, Span, kw, sym};
use rustc_trait_selection::infer::InferCtxtExt;
use tracing::debug;

use crate::FnCtxt;
use crate::method::probe::{self, Pick};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(super) fn lint_edition_dependent_dot_call(
        &self,
        self_ty: Ty<'tcx>,
        segment: &hir::PathSegment<'_>,
        span: Span,
        call_expr: &'tcx hir::Expr<'tcx>,
        self_expr: &'tcx hir::Expr<'tcx>,
        pick: &Pick<'tcx>,
        args: &'tcx [hir::Expr<'tcx>],
    ) {
        debug!(
            "lookup(method_name={}, self_ty={:?}, call_expr={:?}, self_expr={:?})",
            segment.ident, self_ty, call_expr, self_expr
        );

        let (prelude_or_array_lint, edition) = match segment.ident.name {
            // `try_into` was added to the prelude in Rust 2021.
            sym::try_into if !span.at_least_rust_2021() => (RUST_2021_PRELUDE_COLLISIONS, "2021"),
            // `Future::poll` was added to the prelude in Rust 2024.
            sym::poll
                // We check that the self type is `Pin<&mut _>` to avoid false positives for this common name.
                if !span.at_least_rust_2024()
                    && let ty::Adt(adt_def, args) = self_ty.kind()
                    && self.tcx.is_lang_item(adt_def.did(), hir::LangItem::Pin)
                    && let ty::Ref(_, _, ty::Mutability::Mut) =
                        args[0].as_type().unwrap().kind() =>
            {
                (RUST_2024_PRELUDE_COLLISIONS, "2024")
            }
            // `IntoFuture::into_future` was added to the prelude in Rust 2024.
            sym::into_future if !span.at_least_rust_2024() => {
                (RUST_2024_PRELUDE_COLLISIONS, "2024")
            }
            // `into_iter` wasn't added to the prelude,
            // but `[T; N].into_iter()` doesn't resolve to IntoIterator::into_iter
            // before Rust 2021, which results in the same problem.
            // It is only a problem for arrays.
            sym::into_iter => {
                if let ty::Array(..) = self_ty.kind()
                    && !span.at_least_rust_2021()
                {
                    // In this case, it wasn't really a prelude addition that was the problem.
                    // Instead, the problem is that the array-into_iter hack will no longer
                    // apply in Rust 2021.
                    (ARRAY_INTO_ITER, "2021")
                } else if self_ty.boxed_ty().is_some_and(Ty::is_slice)
                    && !span.at_least_rust_2024()
                {
                    // In this case, it wasn't really a prelude addition that was the problem.
                    // Instead, the problem is that the boxed-slice-into_iter hack will no
                    // longer apply in Rust 2024.
                    (BOXED_SLICE_INTO_ITER, "2024")
                } else {
                    return;
                }
            }
            _ => return,
        };

        // No need to lint if method came from std/core, as that will now be in the prelude
        if STDLIB_STABLE_CRATES.contains(&self.tcx.crate_name(pick.item.def_id.krate)) {
            return;
        }

        if matches!(pick.kind, probe::PickKind::InherentImplPick | probe::PickKind::ObjectPick) {
            // avoid repeatedly adding unneeded `&*`s
            if pick.autoderefs == 1
                && matches!(
                    pick.autoref_or_ptr_adjustment,
                    Some(probe::AutorefOrPtrAdjustment::Autoref { .. })
                )
                && matches!(self_ty.kind(), ty::Ref(..))
            {
                return;
            }

            // if it's an inherent `self` method (not `&self` or `&mut self`), it will take
            // precedence over the `TryInto` impl, and thus won't break in 2021 edition
            if pick.autoderefs == 0 && pick.autoref_or_ptr_adjustment.is_none() {
                return;
            }

            // Inherent impls only require not relying on autoref and autoderef in order to
            // ensure that the trait implementation won't be used
            self.tcx.node_span_lint(
                prelude_or_array_lint,
                self_expr.hir_id,
                self_expr.span,
                |lint| {
                    lint.primary_message(format!(
                        "trait method `{}` will become ambiguous in Rust {edition}",
                        segment.ident.name
                    ));

                    let sp = self_expr.span;

                    let derefs = "*".repeat(pick.autoderefs);

                    let autoref = match pick.autoref_or_ptr_adjustment {
                        Some(probe::AutorefOrPtrAdjustment::Autoref { mutbl, .. }) => {
                            mutbl.ref_prefix_str()
                        }
                        Some(probe::AutorefOrPtrAdjustment::ToConstPtr) | None => "",
                        Some(probe::AutorefOrPtrAdjustment::ReborrowPin(mutbl)) => match mutbl {
                            hir::Mutability::Mut => "Pin<&mut ",
                            hir::Mutability::Not => "Pin<&",
                        },
                    };
                    if let Ok(self_expr) = self.sess().source_map().span_to_snippet(self_expr.span)
                    {
                        let mut self_adjusted =
                            if let Some(probe::AutorefOrPtrAdjustment::ToConstPtr) =
                                pick.autoref_or_ptr_adjustment
                            {
                                format!("{derefs}{self_expr} as *const _")
                            } else {
                                format!("{autoref}{derefs}{self_expr}")
                            };

                        if let Some(probe::AutorefOrPtrAdjustment::ReborrowPin(_)) =
                            pick.autoref_or_ptr_adjustment
                        {
                            self_adjusted.push('>');
                        }

                        lint.span_suggestion(
                            sp,
                            "disambiguate the method call",
                            format!("({self_adjusted})"),
                            Applicability::MachineApplicable,
                        );
                    } else {
                        let self_adjusted = if let Some(probe::AutorefOrPtrAdjustment::ToConstPtr) =
                            pick.autoref_or_ptr_adjustment
                        {
                            format!("{derefs}(...) as *const _")
                        } else {
                            format!("{autoref}{derefs}...")
                        };
                        lint.span_help(
                            sp,
                            format!("disambiguate the method call with `({self_adjusted})`",),
                        );
                    }
                },
            );
        } else {
            // trait implementations require full disambiguation to not clash with the new prelude
            // additions (i.e. convert from dot-call to fully-qualified call)
            self.tcx.node_span_lint(
                prelude_or_array_lint,
                call_expr.hir_id,
                call_expr.span,
                |lint| {
                    lint.primary_message(format!(
                        "trait method `{}` will become ambiguous in Rust {edition}",
                        segment.ident.name
                    ));

                    let sp = call_expr.span;
                    let trait_name = self.trait_path_or_bare_name(
                        span,
                        call_expr.hir_id,
                        pick.item.container_id(self.tcx),
                    );

                    let (self_adjusted, precise) = self.adjust_expr(pick, self_expr, sp);
                    if precise {
                        let args = args.iter().fold(String::new(), |mut string, arg| {
                            let span = arg.span.find_ancestor_inside(sp).unwrap_or_default();
                            write!(
                                string,
                                ", {}",
                                self.sess().source_map().span_to_snippet(span).unwrap()
                            )
                            .unwrap();
                            string
                        });

                        lint.span_suggestion(
                            sp,
                            "disambiguate the associated function",
                            format!(
                                "{}::{}{}({}{})",
                                trait_name,
                                segment.ident.name,
                                if let Some(args) = segment.args.as_ref().and_then(|args| self
                                    .sess()
                                    .source_map()
                                    .span_to_snippet(args.span_ext)
                                    .ok())
                                {
                                    // Keep turbofish.
                                    format!("::{args}")
                                } else {
                                    String::new()
                                },
                                self_adjusted,
                                args,
                            ),
                            Applicability::MachineApplicable,
                        );
                    } else {
                        lint.span_help(
                            sp,
                            format!(
                                "disambiguate the associated function with `{}::{}(...)`",
                                trait_name, segment.ident,
                            ),
                        );
                    }
                },
            );
        }
    }

    pub(super) fn lint_fully_qualified_call_from_2018(
        &self,
        span: Span,
        method_name: Ident,
        self_ty: Ty<'tcx>,
        self_ty_span: Span,
        expr_id: hir::HirId,
        pick: &Pick<'tcx>,
    ) {
        // Rust 2021 and later is already using the new prelude
        if span.at_least_rust_2021() {
            return;
        }

        // These are the fully qualified methods added to prelude in Rust 2021
        if !matches!(method_name.name, sym::try_into | sym::try_from | sym::from_iter) {
            return;
        }

        // No need to lint if method came from std/core, as that will now be in the prelude
        if STDLIB_STABLE_CRATES.contains(&self.tcx.crate_name(pick.item.def_id.krate)) {
            return;
        }

        // For from_iter, check if the type actually implements FromIterator.
        // If we know it does not, we don't need to warn.
        if method_name.name == sym::from_iter {
            if let Some(trait_def_id) = self.tcx.get_diagnostic_item(sym::FromIterator) {
                let any_type = self.infcx.next_ty_var(span);
                if !self
                    .infcx
                    .type_implements_trait(trait_def_id, [self_ty, any_type], self.param_env)
                    .may_apply()
                {
                    return;
                }
            }
        }

        // No need to lint if this is an inherent method called on a specific type, like `Vec::foo(...)`,
        // since such methods take precedence over trait methods.
        if matches!(pick.kind, probe::PickKind::InherentImplPick) {
            return;
        }

        self.tcx.node_span_lint(RUST_2021_PRELUDE_COLLISIONS, expr_id, span, |lint| {
            lint.primary_message(format!(
                "trait-associated function `{}` will become ambiguous in Rust 2021",
                method_name.name
            ));

            // "type" refers to either a type or, more likely, a trait from which
            // the associated function or method is from.
            let container_id = pick.item.container_id(self.tcx);
            let trait_path = self.trait_path_or_bare_name(span, expr_id, container_id);
            let trait_generics = self.tcx.generics_of(container_id);

            let trait_name =
                if trait_generics.own_params.len() <= trait_generics.has_self as usize {
                    trait_path
                } else {
                    let counts = trait_generics.own_counts();
                    format!(
                        "{}<{}>",
                        trait_path,
                        std::iter::repeat("'_")
                            .take(counts.lifetimes)
                            .chain(std::iter::repeat("_").take(
                                counts.types + counts.consts - trait_generics.has_self as usize
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };

            let mut self_ty_name = self_ty_span
                .find_ancestor_inside(span)
                .and_then(|span| self.sess().source_map().span_to_snippet(span).ok())
                .unwrap_or_else(|| self_ty.to_string());

            // Get the number of generics the self type has (if an Adt) unless we can determine that
            // the user has written the self type with generics already which we (naively) do by looking
            // for a "<" in `self_ty_name`.
            if !self_ty_name.contains('<') {
                if let ty::Adt(def, _) = self_ty.kind() {
                    let generics = self.tcx.generics_of(def.did());
                    if !generics.is_own_empty() {
                        let counts = generics.own_counts();
                        self_ty_name += &format!(
                            "<{}>",
                            std::iter::repeat("'_")
                                .take(counts.lifetimes)
                                .chain(std::iter::repeat("_").take(counts.types + counts.consts))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                }
            }
            lint.span_suggestion(
                span,
                "disambiguate the associated function",
                format!("<{} as {}>::{}", self_ty_name, trait_name, method_name.name,),
                Applicability::MachineApplicable,
            );
        });
    }

    fn trait_path_or_bare_name(
        &self,
        span: Span,
        expr_hir_id: HirId,
        trait_def_id: DefId,
    ) -> String {
        self.trait_path(span, expr_hir_id, trait_def_id).unwrap_or_else(|| {
            let key = self.tcx.def_key(trait_def_id);
            format!("{}", key.disambiguated_data.data)
        })
    }

    fn trait_path(&self, span: Span, expr_hir_id: HirId, trait_def_id: DefId) -> Option<String> {
        let applicable_traits = self.tcx.in_scope_traits(expr_hir_id)?;
        let applicable_trait = applicable_traits.iter().find(|t| t.def_id == trait_def_id)?;
        if applicable_trait.import_ids.is_empty() {
            // The trait was declared within the module, we only need to use its name.
            return None;
        }

        let import_items: Vec<_> = applicable_trait
            .import_ids
            .iter()
            .map(|&import_id| self.tcx.hir_expect_item(import_id))
            .collect();

        // Find an identifier with which this trait was imported (note that `_` doesn't count).
        for item in import_items.iter() {
            let (_, kind) = item.expect_use();
            match kind {
                hir::UseKind::Single(ident) => {
                    if ident.name != kw::Underscore {
                        return Some(format!("{}", ident.name));
                    }
                }
                hir::UseKind::Glob => return None, // Glob import, so just use its name.
                hir::UseKind::ListStem => unreachable!(),
            }
        }

        // All that is left is `_`! We need to use the full path. It doesn't matter which one we
        // pick, so just take the first one.
        match import_items[0].kind {
            ItemKind::Use(path, _) => {
                Some(join_path_idents(path.segments.iter().map(|seg| seg.ident)))
            }
            _ => {
                span_bug!(span, "unexpected item kind, expected a use: {:?}", import_items[0].kind);
            }
        }
    }

    /// Creates a string version of the `expr` that includes explicit adjustments.
    /// Returns the string and also a bool indicating whether this is a *precise*
    /// suggestion.
    fn adjust_expr(
        &self,
        pick: &Pick<'tcx>,
        expr: &hir::Expr<'tcx>,
        outer: Span,
    ) -> (String, bool) {
        let derefs = "*".repeat(pick.autoderefs);

        let autoref = match pick.autoref_or_ptr_adjustment {
            Some(probe::AutorefOrPtrAdjustment::Autoref { mutbl, .. }) => mutbl.ref_prefix_str(),
            Some(probe::AutorefOrPtrAdjustment::ToConstPtr) | None => "",
            Some(probe::AutorefOrPtrAdjustment::ReborrowPin(mutbl)) => match mutbl {
                hir::Mutability::Mut => "Pin<&mut ",
                hir::Mutability::Not => "Pin<&",
            },
        };

        let (expr_text, precise) = if let Some(expr_text) = expr
            .span
            .find_ancestor_inside(outer)
            .and_then(|span| self.sess().source_map().span_to_snippet(span).ok())
        {
            (expr_text, true)
        } else {
            ("(..)".to_string(), false)
        };

        let mut adjusted_text = if let Some(probe::AutorefOrPtrAdjustment::ToConstPtr) =
            pick.autoref_or_ptr_adjustment
        {
            format!("{derefs}{expr_text} as *const _")
        } else {
            format!("{autoref}{derefs}{expr_text}")
        };

        if let Some(probe::AutorefOrPtrAdjustment::ReborrowPin(_)) = pick.autoref_or_ptr_adjustment
        {
            adjusted_text.push('>');
        }

        (adjusted_text, precise)
    }
}
