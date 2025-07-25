use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_from_proc_macro;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::SpanRangeExt;
use hir::def_id::DefId;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{ExprKind, Item, ItemKind, QPath, UseKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::symbol::kw;
use rustc_span::{Symbol, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `<integer>::max_value()`, `std::<integer>::MAX`,
    /// `std::<float>::EPSILON`, etc.
    ///
    /// ### Why is this bad?
    /// All of these have been superseded by the associated constants on their respective types,
    /// such as `i128::MAX`. These legacy items may be deprecated in a future version of rust.
    ///
    /// ### Example
    /// ```rust
    /// let eps = std::f32::EPSILON;
    /// ```
    /// Use instead:
    /// ```rust
    /// let eps = f32::EPSILON;
    /// ```
    #[clippy::version = "1.79.0"]
    pub LEGACY_NUMERIC_CONSTANTS,
    style,
    "checks for usage of legacy std numeric constants and methods"
}
pub struct LegacyNumericConstants {
    msrv: Msrv,
}

impl LegacyNumericConstants {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(LegacyNumericConstants => [LEGACY_NUMERIC_CONSTANTS]);

impl<'tcx> LateLintPass<'tcx> for LegacyNumericConstants {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        // Integer modules are "TBD" deprecated, and the contents are too,
        // so lint on the `use` statement directly.
        if let ItemKind::Use(path, kind @ (UseKind::Single(_) | UseKind::Glob)) = item.kind
            && !item.span.in_external_macro(cx.sess().source_map())
            // use `present_items` because it could be in either type_ns or value_ns
            && let Some(res) = path.res.present_items().next()
            && let Some(def_id) = res.opt_def_id()
            && self.msrv.meets(cx, msrvs::NUMERIC_ASSOCIATED_CONSTANTS)
        {
            let module = if is_integer_module(cx, def_id) {
                true
            } else if is_numeric_const(cx, def_id) {
                false
            } else {
                return;
            };

            span_lint_and_then(
                cx,
                LEGACY_NUMERIC_CONSTANTS,
                path.span,
                if module {
                    "importing legacy numeric constants"
                } else {
                    "importing a legacy numeric constant"
                },
                |diag| {
                    if let UseKind::Single(ident) = kind
                        && ident.name == kw::Underscore
                    {
                        diag.help("remove this import");
                        return;
                    }

                    let def_path = cx.get_def_path(def_id);

                    if module && let [.., module_name] = &*def_path {
                        if kind == UseKind::Glob {
                            diag.help(format!("remove this import and use associated constants `{module_name}::<CONST>` from the primitive type instead"));
                        } else {
                            diag.help("remove this import").note(format!(
                                "then `{module_name}::<CONST>` will resolve to the respective associated constant"
                            ));
                        }
                    } else if let [.., module_name, name] = &*def_path {
                        diag.help(
                            format!("remove this import and use the associated constant `{module_name}::{name}` from the primitive type instead")
                        );
                    }
                },
            );
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx rustc_hir::Expr<'tcx>) {
        // `std::<integer>::<CONST>` check
        let (sugg, msg) = if let ExprKind::Path(qpath) = &expr.kind
            && let QPath::Resolved(None, path) = qpath
            && let Some(def_id) = path.res.opt_def_id()
            && is_numeric_const(cx, def_id)
            && let [.., mod_name, name] = &*cx.get_def_path(def_id)
            // Skip linting if this usage looks identical to the associated constant,
            // since this would only require removing a `use` import (which is already linted).
            && !is_numeric_const_path_canonical(path, [*mod_name, *name])
        {
            (
                vec![(expr.span, format!("{mod_name}::{name}"))],
                "usage of a legacy numeric constant",
            )
        // `<integer>::xxx_value` check
        } else if let ExprKind::Call(func, []) = &expr.kind
            && let ExprKind::Path(qpath) = &func.kind
            && let QPath::TypeRelative(ty, last_segment) = qpath
            && let Some(def_id) = cx.qpath_res(qpath, func.hir_id).opt_def_id()
            && is_integer_method(cx, def_id)
        {
            let mut sugg = vec![
                // Replace the function name up to the end by the constant name
                (
                    last_segment.ident.span.to(expr.span.shrink_to_hi()),
                    last_segment.ident.name.as_str()[..=2].to_ascii_uppercase(),
                ),
            ];
            let before_span = expr.span.shrink_to_lo().until(ty.span);
            if !before_span.is_empty() {
                // Remove everything before the type name
                sugg.push((before_span, String::new()));
            }
            // Use `::` between the type name and the constant
            let between_span = ty.span.shrink_to_hi().until(last_segment.ident.span);
            if !between_span.check_source_text(cx, |s| s == "::") {
                sugg.push((between_span, String::from("::")));
            }
            (sugg, "usage of a legacy numeric method")
        } else {
            return;
        };

        if !expr.span.in_external_macro(cx.sess().source_map())
            && self.msrv.meets(cx, msrvs::NUMERIC_ASSOCIATED_CONSTANTS)
            && !is_from_proc_macro(cx, expr)
        {
            span_lint_and_then(cx, LEGACY_NUMERIC_CONSTANTS, expr.span, msg, |diag| {
                diag.multipart_suggestion_verbose(
                    "use the associated constant instead",
                    sugg,
                    Applicability::MaybeIncorrect,
                );
            });
        }
    }
}

fn is_integer_module(cx: &LateContext<'_>, did: DefId) -> bool {
    matches!(
        cx.tcx.get_diagnostic_name(did),
        Some(
            sym::isize_legacy_mod
                | sym::i128_legacy_mod
                | sym::i64_legacy_mod
                | sym::i32_legacy_mod
                | sym::i16_legacy_mod
                | sym::i8_legacy_mod
                | sym::usize_legacy_mod
                | sym::u128_legacy_mod
                | sym::u64_legacy_mod
                | sym::u32_legacy_mod
                | sym::u16_legacy_mod
                | sym::u8_legacy_mod
        )
    )
}

fn is_numeric_const(cx: &LateContext<'_>, did: DefId) -> bool {
    matches!(
        cx.tcx.get_diagnostic_name(did),
        Some(
            sym::isize_legacy_const_max
                | sym::isize_legacy_const_min
                | sym::i128_legacy_const_max
                | sym::i128_legacy_const_min
                | sym::i16_legacy_const_max
                | sym::i16_legacy_const_min
                | sym::i32_legacy_const_max
                | sym::i32_legacy_const_min
                | sym::i64_legacy_const_max
                | sym::i64_legacy_const_min
                | sym::i8_legacy_const_max
                | sym::i8_legacy_const_min
                | sym::usize_legacy_const_max
                | sym::usize_legacy_const_min
                | sym::u128_legacy_const_max
                | sym::u128_legacy_const_min
                | sym::u16_legacy_const_max
                | sym::u16_legacy_const_min
                | sym::u32_legacy_const_max
                | sym::u32_legacy_const_min
                | sym::u64_legacy_const_max
                | sym::u64_legacy_const_min
                | sym::u8_legacy_const_max
                | sym::u8_legacy_const_min
                | sym::f32_legacy_const_digits
                | sym::f32_legacy_const_epsilon
                | sym::f32_legacy_const_infinity
                | sym::f32_legacy_const_mantissa_dig
                | sym::f32_legacy_const_max
                | sym::f32_legacy_const_max_10_exp
                | sym::f32_legacy_const_max_exp
                | sym::f32_legacy_const_min
                | sym::f32_legacy_const_min_10_exp
                | sym::f32_legacy_const_min_exp
                | sym::f32_legacy_const_min_positive
                | sym::f32_legacy_const_nan
                | sym::f32_legacy_const_neg_infinity
                | sym::f32_legacy_const_radix
                | sym::f64_legacy_const_digits
                | sym::f64_legacy_const_epsilon
                | sym::f64_legacy_const_infinity
                | sym::f64_legacy_const_mantissa_dig
                | sym::f64_legacy_const_max
                | sym::f64_legacy_const_max_10_exp
                | sym::f64_legacy_const_max_exp
                | sym::f64_legacy_const_min
                | sym::f64_legacy_const_min_10_exp
                | sym::f64_legacy_const_min_exp
                | sym::f64_legacy_const_min_positive
                | sym::f64_legacy_const_nan
                | sym::f64_legacy_const_neg_infinity
                | sym::f64_legacy_const_radix
        )
    )
}

// Whether path expression looks like `i32::MAX`
fn is_numeric_const_path_canonical(expr_path: &hir::Path<'_>, [mod_name, name]: [Symbol; 2]) -> bool {
    let [
        hir::PathSegment {
            ident: one, args: None, ..
        },
        hir::PathSegment {
            ident: two, args: None, ..
        },
    ] = expr_path.segments
    else {
        return false;
    };

    one.name == mod_name && two.name == name
}

fn is_integer_method(cx: &LateContext<'_>, did: DefId) -> bool {
    matches!(
        cx.tcx.get_diagnostic_name(did),
        Some(
            sym::isize_legacy_fn_max_value
                | sym::isize_legacy_fn_min_value
                | sym::i128_legacy_fn_max_value
                | sym::i128_legacy_fn_min_value
                | sym::i16_legacy_fn_max_value
                | sym::i16_legacy_fn_min_value
                | sym::i32_legacy_fn_max_value
                | sym::i32_legacy_fn_min_value
                | sym::i64_legacy_fn_max_value
                | sym::i64_legacy_fn_min_value
                | sym::i8_legacy_fn_max_value
                | sym::i8_legacy_fn_min_value
                | sym::usize_legacy_fn_max_value
                | sym::usize_legacy_fn_min_value
                | sym::u128_legacy_fn_max_value
                | sym::u128_legacy_fn_min_value
                | sym::u16_legacy_fn_max_value
                | sym::u16_legacy_fn_min_value
                | sym::u32_legacy_fn_max_value
                | sym::u32_legacy_fn_min_value
                | sym::u64_legacy_fn_max_value
                | sym::u64_legacy_fn_min_value
                | sym::u8_legacy_fn_max_value
                | sym::u8_legacy_fn_min_value
        )
    )
}
