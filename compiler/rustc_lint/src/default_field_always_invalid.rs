use rustc_hir as hir;
use rustc_middle::lint::LintLevelSource;
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_session::lint::Level;
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::DefaultFieldAlwaysInvalidConst;
use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `default_field_always_invalid_const` lint checks for structs with
    /// default fields const values that will *always* fail to be created.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![feature(default_field_values)]
    /// #[deny(default_field_always_invalid_const)]
    /// struct Foo {
    ///     bar: u8 = 130 + 130, // `260` doesn't fit in `u8`
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Without this lint, the error would only happen only during construction
    /// of the affected type. For example, given the type above, `Foo { .. }`
    /// would always fail to build, but `Foo { bar: 0 }` would be accepted. This
    /// lint will catch accidental cases of const values that would fail to
    /// compile, but won't detect cases that are only partially evaluated.
    pub DEFAULT_FIELD_ALWAYS_INVALID_CONST,
    Deny,
    "using this default field will always fail to compile"
}

declare_lint_pass!(DefaultFieldAlwaysInvalid => [DEFAULT_FIELD_ALWAYS_INVALID_CONST]);

impl<'tcx> LateLintPass<'tcx> for DefaultFieldAlwaysInvalid {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        let data = match item.kind {
            hir::ItemKind::Struct(data, _generics) => data,
            _ => return,
        };
        let hir::VariantData::Struct { fields, recovered: _ } = data else {
            return;
        };

        let (level, source) =
            cx.tcx.lint_level_at_node(DEFAULT_FIELD_ALWAYS_INVALID_CONST, item.hir_id());
        match level {
            Level::Deny | Level::Forbid => {}
            Level::Warn | Level::ForceWarn(_) | Level::Expect(_) => {
                // We *can't* turn the const eval error into a warning, so we make it a
                // warning to not use `#[warn(default_field_always_invalid_const)]`.
                let invalid_msg = "lint `default_field_always_invalid_const` can't be warned on";
                #[allow(rustc::diagnostic_outside_of_impl, rustc::untranslatable_diagnostic)]
                if let LintLevelSource::Node { span, .. } = source {
                    let mut err = cx.tcx.sess.dcx().struct_span_warn(span, invalid_msg);
                    err.span_label(
                        span,
                        "either `deny` or `allow`, no other lint level is supported for this lint",
                    );
                    err.emit();
                } else {
                    cx.tcx.sess.dcx().warn(invalid_msg);
                }
            }
            Level::Allow => {
                // We don't even look at the fields.
                return;
            }
        }
        for field in fields {
            if let Some(c) = field.default
                && let Some(_ty) = cx.tcx.type_of(c.def_id).no_bound_vars()
                && let Err(ErrorHandled::Reported(_, _)) = cx.tcx.const_eval_poly(c.def_id.into())
            {
                // We use the item's hir id because the const's hir id might resolve inside of a
                // foreign macro, meaning the lint won't trigger.
                cx.tcx.emit_node_span_lint(
                    DEFAULT_FIELD_ALWAYS_INVALID_CONST,
                    item.hir_id(),
                    field.span,
                    DefaultFieldAlwaysInvalidConst { span: field.span, help: () },
                );
            }
        }
    }
}
