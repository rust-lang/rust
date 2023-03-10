use rustc_ast::{
    ast,
    visit::{FnCtxt, FnKind},
};
use rustc_errors::DecorateLint;
use rustc_session::lint::builtin::UNSAFE_OP_IN_UNSAFE_FN;
use rustc_span::{sym, Span};

use crate::{lints::BuiltinUnsafe, EarlyContext, EarlyLintPass, LintContext};

declare_lint! {
    /// The `unsafe_obligation_define` lint triggers when an "unsafe contract"
    /// is defined.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unsafe_obligation_define)]
    /// unsafe trait Foo {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// An "unsafe contract" is a set of invariants which must be upheld in
    /// order to prevent Undefined Behaviour in unsafe code. This lint triggers
    /// when such a contract is defined, for example when defining an
    /// `unsafe trait` or unsafe trait method without a body.
    pub UNSAFE_OBLIGATION_DEFINE,
    Allow,
    "definition of unsafe contract"
}

declare_lint! {
    /// The `unsafe_obligation_discharge` lint triggers when an
    /// "unsafe contract"'s invariants are consumed.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unsafe_obligation_discharge)]
    /// fn main() {
    ///     unsafe {
    ///
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// An "unsafe contract" is a set of invariants which must be upheld in
    /// order to prevent Undefined Behaviour in unsafe code. This lint triggers
    /// when such a contract's invariants must be upheld. For example, `unsafe`
    /// blocks may call functions which have safety variants which must be
    /// upheld.
    pub UNSAFE_OBLIGATION_DISCHARGE,
    Allow,
    "discharge of unsafe responsibilities"
}

declare_lint_pass!(UnsafeCode => [UNSAFE_OBLIGATION_DEFINE, UNSAFE_OBLIGATION_DISCHARGE]);

enum ObligationKind {
    Discharge,
    Define,
}

impl UnsafeCode {
    fn report_unsafe(
        &self,
        cx: &EarlyContext<'_>,
        span: Span,
        decorate: impl for<'a> DecorateLint<'a, ()>,
        kind: ObligationKind,
    ) {
        // This comes from a macro that has `#[allow_internal_unsafe]`.
        if span.allows_unsafe() {
            return;
        }

        cx.emit_spanned_lint(
            match kind {
                ObligationKind::Discharge => UNSAFE_OBLIGATION_DISCHARGE,
                ObligationKind::Define => UNSAFE_OBLIGATION_DEFINE,
            },
            span,
            decorate,
        );
    }
}

impl EarlyLintPass for UnsafeCode {
    fn check_attribute(&mut self, cx: &EarlyContext<'_>, attr: &ast::Attribute) {
        if attr.has_name(sym::allow_internal_unsafe) {
            self.report_unsafe(
                cx,
                attr.span,
                BuiltinUnsafe::AllowInternalUnsafe,
                ObligationKind::Discharge,
            );
        }
    }

    #[inline]
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        if let ast::ExprKind::Block(ref blk, _) = e.kind {
            // Don't warn about generated blocks; that'll just pollute the output.
            if blk.rules == ast::BlockCheckMode::Unsafe(ast::UserProvided) {
                self.report_unsafe(
                    cx,
                    blk.span,
                    BuiltinUnsafe::UnsafeBlock,
                    ObligationKind::Discharge,
                );
            }
        }
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, it: &ast::Item) {
        match it.kind {
            ast::ItemKind::Trait(box ast::Trait { unsafety: ast::Unsafe::Yes(_), .. }) => {
                self.report_unsafe(cx, it.span, BuiltinUnsafe::UnsafeTrait, ObligationKind::Define);
            }

            ast::ItemKind::Impl(box ast::Impl { unsafety: ast::Unsafe::Yes(_), .. }) => {
                self.report_unsafe(
                    cx,
                    it.span,
                    BuiltinUnsafe::UnsafeImpl,
                    ObligationKind::Discharge,
                );
            }

            ast::ItemKind::Fn(..) => {
                if let Some(attr) = cx.sess().find_by_name(&it.attrs, sym::no_mangle) {
                    self.report_unsafe(
                        cx,
                        attr.span,
                        BuiltinUnsafe::NoMangleFn,
                        ObligationKind::Discharge,
                    );
                }

                if let Some(attr) = cx.sess().find_by_name(&it.attrs, sym::export_name) {
                    self.report_unsafe(
                        cx,
                        attr.span,
                        BuiltinUnsafe::ExportNameFn,
                        ObligationKind::Discharge,
                    );
                }

                if let Some(attr) = cx.sess().find_by_name(&it.attrs, sym::link_section) {
                    self.report_unsafe(
                        cx,
                        attr.span,
                        BuiltinUnsafe::LinkSectionFn,
                        ObligationKind::Discharge,
                    );
                }
            }

            ast::ItemKind::Static(..) => {
                if let Some(attr) = cx.sess().find_by_name(&it.attrs, sym::no_mangle) {
                    self.report_unsafe(
                        cx,
                        attr.span,
                        BuiltinUnsafe::NoMangleStatic,
                        ObligationKind::Discharge,
                    );
                }

                if let Some(attr) = cx.sess().find_by_name(&it.attrs, sym::export_name) {
                    self.report_unsafe(
                        cx,
                        attr.span,
                        BuiltinUnsafe::ExportNameStatic,
                        ObligationKind::Discharge,
                    );
                }

                if let Some(attr) = cx.sess().find_by_name(&it.attrs, sym::link_section) {
                    self.report_unsafe(
                        cx,
                        attr.span,
                        BuiltinUnsafe::LinkSectionStatic,
                        ObligationKind::Discharge,
                    );
                }
            }

            _ => {}
        }
    }

    fn check_impl_item(&mut self, cx: &EarlyContext<'_>, it: &ast::AssocItem) {
        if let ast::AssocItemKind::Fn(..) = it.kind {
            if let Some(attr) = cx.sess().find_by_name(&it.attrs, sym::no_mangle) {
                self.report_unsafe(
                    cx,
                    attr.span,
                    BuiltinUnsafe::NoMangleMethod,
                    ObligationKind::Discharge,
                );
            }
            if let Some(attr) = cx.sess().find_by_name(&it.attrs, sym::export_name) {
                self.report_unsafe(
                    cx,
                    attr.span,
                    BuiltinUnsafe::ExportNameMethod,
                    ObligationKind::Discharge,
                );
            }
        }
    }

    fn check_fn(&mut self, cx: &EarlyContext<'_>, fk: FnKind<'_>, span: Span, _: ast::NodeId) {
        if let FnKind::Fn(
            ctxt,
            _,
            ast::FnSig { header: ast::FnHeader { unsafety: ast::Unsafe::Yes(_), .. }, .. },
            _,
            _,
            body,
        ) = fk
        {
            let decorator = match ctxt {
                FnCtxt::Foreign => return,
                FnCtxt::Free => BuiltinUnsafe::DeclUnsafeFn,
                FnCtxt::Assoc(_) if body.is_none() => {
                    // there is no body, so we know that it cannot contain
                    // unsafety which does more than simply define an unsafety
                    // contract, see below.
                    return self.report_unsafe(
                        cx,
                        span,
                        BuiltinUnsafe::DeclUnsafeMethod,
                        ObligationKind::Define,
                    );
                }
                FnCtxt::Assoc(_) => BuiltinUnsafe::ImplUnsafeMethod,
            };

            // Unsafe methods can merely define unsafety contracts, but they
            // also give free rein for the body of the function to contain
            // unsafe code which is not necessarily covered by said contract.
            // If unsafe code in the function body is allowed without unsafe
            // blocks, then it is just a regular discharge of unsafe
            // responibilties.
            if cx.get_lint_level(UNSAFE_OP_IN_UNSAFE_FN)
                >= cx.get_lint_level(UNSAFE_OBLIGATION_DEFINE)
            {
                self.report_unsafe(cx, span, decorator, ObligationKind::Define);
            } else {
                self.report_unsafe(cx, span, decorator, ObligationKind::Discharge);
            }
        }
    }
}
