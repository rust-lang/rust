use hir::def_id::DefId;
use hir::HirId;
use hir::ItemKind;
use rustc_ast::Mutability;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::{Adt, Array, Ref, Ty};
use rustc_session::lint::builtin::RUST_2021_PRELUDE_COLLISIONS;
use rustc_span::symbol::kw::{Empty, Underscore};
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;
use rustc_trait_selection::infer::InferCtxtExt;

use crate::check::{
    method::probe::{self, Pick},
    FnCtxt,
};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(super) fn lint_dot_call_from_2018(
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

        // Rust 2021 and later is already using the new prelude
        if span.rust_2021() {
            return;
        }

        let prelude_or_array_lint = match segment.ident.name {
            // `try_into` was added to the prelude in Rust 2021.
            sym::try_into => RUST_2021_PRELUDE_COLLISIONS,
            // `into_iter` wasn't added to the prelude,
            // but `[T; N].into_iter()` doesn't resolve to IntoIterator::into_iter
            // before Rust 2021, which results in the same problem.
            // It is only a problem for arrays.
            sym::into_iter if let Array(..) = self_ty.kind() => {
                // In this case, it wasn't really a prelude addition that was the problem.
                // Instead, the problem is that the array-into_iter hack will no longer apply in Rust 2021.
                rustc_lint::ARRAY_INTO_ITER
            }
            _ => return,
        };

        // No need to lint if method came from std/core, as that will now be in the prelude
        if matches!(self.tcx.crate_name(pick.item.def_id.krate), sym::std | sym::core) {
            return;
        }

        if matches!(pick.kind, probe::PickKind::InherentImplPick | probe::PickKind::ObjectPick) {
            // avoid repeatedly adding unneeded `&*`s
            if pick.autoderefs == 1
                && matches!(
                    pick.autoref_or_ptr_adjustment,
                    Some(probe::AutorefOrPtrAdjustment::Autoref { .. })
                )
                && matches!(self_ty.kind(), Ref(..))
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
            self.tcx.struct_span_lint_hir(
                prelude_or_array_lint,
                self_expr.hir_id,
                self_expr.span,
                |lint| {
                    let sp = self_expr.span;

                    let mut lint = lint.build(&format!(
                        "trait method `{}` will become ambiguous in Rust 2021",
                        segment.ident.name
                    ));

                    let derefs = "*".repeat(pick.autoderefs);

                    let autoref = match pick.autoref_or_ptr_adjustment {
                        Some(probe::AutorefOrPtrAdjustment::Autoref {
                            mutbl: Mutability::Mut,
                            ..
                        }) => "&mut ",
                        Some(probe::AutorefOrPtrAdjustment::Autoref {
                            mutbl: Mutability::Not,
                            ..
                        }) => "&",
                        Some(probe::AutorefOrPtrAdjustment::ToConstPtr) | None => "",
                    };
                    if let Ok(self_expr) = self.sess().source_map().span_to_snippet(self_expr.span)
                    {
                        let self_adjusted = if let Some(probe::AutorefOrPtrAdjustment::ToConstPtr) =
                            pick.autoref_or_ptr_adjustment
                        {
                            format!("{}{} as *const _", derefs, self_expr)
                        } else {
                            format!("{}{}{}", autoref, derefs, self_expr)
                        };

                        lint.span_suggestion(
                            sp,
                            "disambiguate the method call",
                            format!("({})", self_adjusted),
                            Applicability::MachineApplicable,
                        );
                    } else {
                        let self_adjusted = if let Some(probe::AutorefOrPtrAdjustment::ToConstPtr) =
                            pick.autoref_or_ptr_adjustment
                        {
                            format!("{}(...) as *const _", derefs)
                        } else {
                            format!("{}{}...", autoref, derefs)
                        };
                        lint.span_help(
                            sp,
                            &format!("disambiguate the method call with `({})`", self_adjusted,),
                        );
                    }

                    lint.emit();
                },
            );
        } else {
            // trait implementations require full disambiguation to not clash with the new prelude
            // additions (i.e. convert from dot-call to fully-qualified call)
            self.tcx.struct_span_lint_hir(
                prelude_or_array_lint,
                call_expr.hir_id,
                call_expr.span,
                |lint| {
                    let sp = call_expr.span;
                    let trait_name = self.trait_path_or_bare_name(
                        span,
                        call_expr.hir_id,
                        pick.item.container.id(),
                    );

                    let mut lint = lint.build(&format!(
                        "trait method `{}` will become ambiguous in Rust 2021",
                        segment.ident.name
                    ));

                    let (self_adjusted, precise) = self.adjust_expr(pick, self_expr, sp);
                    if precise {
                        let args = args
                            .iter()
                            .skip(1)
                            .map(|arg| {
                                let span = arg.span.find_ancestor_inside(sp).unwrap_or_default();
                                format!(
                                    ", {}",
                                    self.sess().source_map().span_to_snippet(span).unwrap()
                                )
                            })
                            .collect::<String>();

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
                                    format!("::{}", args)
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
                            &format!(
                                "disambiguate the associated function with `{}::{}(...)`",
                                trait_name, segment.ident,
                            ),
                        );
                    }

                    lint.emit();
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
        if span.rust_2021() {
            return;
        }

        // These are the fully qualified methods added to prelude in Rust 2021
        if !matches!(method_name.name, sym::try_into | sym::try_from | sym::from_iter) {
            return;
        }

        // No need to lint if method came from std/core, as that will now be in the prelude
        if matches!(self.tcx.crate_name(pick.item.def_id.krate), sym::std | sym::core) {
            return;
        }

        // For from_iter, check if the type actually implements FromIterator.
        // If we know it does not, we don't need to warn.
        if method_name.name == sym::from_iter {
            if let Some(trait_def_id) = self.tcx.get_diagnostic_item(sym::FromIterator) {
                if !self
                    .infcx
                    .type_implements_trait(
                        trait_def_id,
                        self_ty,
                        InternalSubsts::empty(),
                        self.param_env,
                    )
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

        self.tcx.struct_span_lint_hir(RUST_2021_PRELUDE_COLLISIONS, expr_id, span, |lint| {
            // "type" refers to either a type or, more likely, a trait from which
            // the associated function or method is from.
            let trait_path = self.trait_path_or_bare_name(span, expr_id, pick.item.container.id());
            let trait_generics = self.tcx.generics_of(pick.item.container.id());

            let trait_name =
                if trait_generics.params.len() <= trait_generics.has_self as usize {
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

            let mut lint = lint.build(&format!(
                "trait-associated function `{}` will become ambiguous in Rust 2021",
                method_name.name
            ));

            let mut self_ty_name = self_ty_span
                .find_ancestor_inside(span)
                .and_then(|span| self.sess().source_map().span_to_snippet(span).ok())
                .unwrap_or_else(|| self_ty.to_string());

            // Get the number of generics the self type has (if an Adt) unless we can determine that
            // the user has written the self type with generics already which we (naively) do by looking
            // for a "<" in `self_ty_name`.
            if !self_ty_name.contains('<') {
                if let Adt(def, _) = self_ty.kind() {
                    let generics = self.tcx.generics_of(def.did);
                    if !generics.params.is_empty() {
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

            lint.emit();
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
            .map(|&import_id| self.tcx.hir().expect_item(import_id))
            .collect();

        // Find an identifier with which this trait was imported (note that `_` doesn't count).
        let any_id = import_items
            .iter()
            .filter_map(|item| if item.ident.name != Underscore { Some(item.ident) } else { None })
            .next();
        if let Some(any_id) = any_id {
            if any_id.name == Empty {
                // Glob import, so just use its name.
                return None;
            } else {
                return Some(format!("{}", any_id));
            }
        }

        // All that is left is `_`! We need to use the full path. It doesn't matter which one we pick,
        // so just take the first one.
        match import_items[0].kind {
            ItemKind::Use(path, _) => Some(
                path.segments
                    .iter()
                    .map(|segment| segment.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::"),
            ),
            _ => {
                span_bug!(span, "unexpected item kind, expected a use: {:?}", import_items[0].kind);
            }
        }
    }

    /// Creates a string version of the `expr` that includes explicit adjustments.
    /// Returns the string and also a bool indicating whther this is a *precise*
    /// suggestion.
    fn adjust_expr(
        &self,
        pick: &Pick<'tcx>,
        expr: &hir::Expr<'tcx>,
        outer: Span,
    ) -> (String, bool) {
        let derefs = "*".repeat(pick.autoderefs);

        let autoref = match pick.autoref_or_ptr_adjustment {
            Some(probe::AutorefOrPtrAdjustment::Autoref { mutbl: Mutability::Mut, .. }) => "&mut ",
            Some(probe::AutorefOrPtrAdjustment::Autoref { mutbl: Mutability::Not, .. }) => "&",
            Some(probe::AutorefOrPtrAdjustment::ToConstPtr) | None => "",
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

        let adjusted_text = if let Some(probe::AutorefOrPtrAdjustment::ToConstPtr) =
            pick.autoref_or_ptr_adjustment
        {
            format!("{}{} as *const _", derefs, expr_text)
        } else {
            format!("{}{}{}", autoref, derefs, expr_text)
        };

        (adjusted_text, precise)
    }
}
