use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::{MaybeDef as _, MaybeTypeckRes as _};
use clippy_utils::{is_lint_allowed, sym};
use core::ptr;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of the `to_be_bytes` method and/or the function `from_be_bytes`.
    ///
    /// ### Why restrict this?
    /// To ensure use of little-endian or the target’s endianness rather than big-endian.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _x = 2i32.to_be_bytes();
    /// let _y = 2i64.to_be_bytes();
    /// ```
    #[clippy::version = "1.72.0"]
    pub BIG_ENDIAN_BYTES,
    restriction,
    "disallows usage of the `to_be_bytes` method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of the `to_ne_bytes` method and/or the function `from_ne_bytes`.
    ///
    /// ### Why restrict this?
    /// To ensure use of explicitly chosen endianness rather than the target’s endianness,
    /// such as when implementing network protocols or file formats rather than FFI.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _x = 2i32.to_ne_bytes();
    /// let _y = 2i64.to_ne_bytes();
    /// ```
    #[clippy::version = "1.72.0"]
    pub HOST_ENDIAN_BYTES,
    restriction,
    "disallows usage of the `to_ne_bytes` method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of the `to_le_bytes` method and/or the function `from_le_bytes`.
    ///
    /// ### Why restrict this?
    /// To ensure use of big-endian or the target’s endianness rather than little-endian.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _x = 2i32.to_le_bytes();
    /// let _y = 2i64.to_le_bytes();
    /// ```
    #[clippy::version = "1.72.0"]
    pub LITTLE_ENDIAN_BYTES,
    restriction,
    "disallows usage of the `to_le_bytes` method"
}

declare_lint_pass!(EndianBytes => [
    BIG_ENDIAN_BYTES,
    HOST_ENDIAN_BYTES,
    LITTLE_ENDIAN_BYTES,
]);

#[derive(Clone, Copy)]
enum Direction {
    From,
    To,
}

impl LateLintPass<'_> for EndianBytes {
    fn check_expr(&mut self, cx: &LateContext<'_>, e: &Expr<'_>) {
        let (sp, direction, lint, msg) = match e.kind {
            // rustfmt wants to break each arm into one line per tuple element which
            // really hurts readability.
            #[rustfmt::skip]
            ExprKind::MethodCall(seg, _, [], _) => match seg.ident.name {
                sym::to_ne_bytes => (seg.ident.span, Direction::To, HOST_ENDIAN_BYTES, "use of `to_ne_bytes`"),
                sym::to_le_bytes => (seg.ident.span, Direction::To, LITTLE_ENDIAN_BYTES, "use of `to_le_bytes`"),
                sym::to_be_bytes => (seg.ident.span, Direction::To, BIG_ENDIAN_BYTES, "use of `to_be_bytes`"),
                _ => return,
            },
            #[rustfmt::skip]
            ExprKind::Path(QPath::TypeRelative(_, seg)) => match seg.ident.name {
                sym::from_ne_bytes => (seg.ident.span, Direction::From, HOST_ENDIAN_BYTES, "use of `from_ne_bytes`"),
                sym::from_le_bytes => (seg.ident.span, Direction::From, LITTLE_ENDIAN_BYTES, "use of `from_le_bytes`"),
                sym::from_be_bytes => (seg.ident.span, Direction::From, BIG_ENDIAN_BYTES, "use of `from_be_bytes`"),
                sym::to_ne_bytes => (seg.ident.span, Direction::To, HOST_ENDIAN_BYTES, "use of `to_ne_bytes`"),
                sym::to_le_bytes => (seg.ident.span, Direction::To, LITTLE_ENDIAN_BYTES, "use of `to_le_bytes`"),
                sym::to_be_bytes => (seg.ident.span, Direction::To, BIG_ENDIAN_BYTES, "use of `to_be_bytes`"),
                _ => return,
            },
            _ => return,
        };
        if let Some(ty) = cx.ty_based_def(e.hir_id).opt_parent(cx).opt_impl_ty(cx)
            && let ty::Uint(_) | ty::Int(_) | ty::Float(_) = *ty.instantiate_identity().skip_normalization().kind()
            // Only check where the name itself comes from. The point of the lints is to
            // catch when the wrong byte order is used so we only care if the current crate
            // decided on the byte order. Which crate actually assembled the path/call
            // isn't relevant for these lints.
            && !sp.in_external_macro(cx.tcx.sess.source_map())
        {
            span_lint_and_then(cx, lint, sp, msg, |diag| {
                if !ptr::addr_eq(lint, HOST_ENDIAN_BYTES) && is_lint_allowed(cx, HOST_ENDIAN_BYTES, e.hir_id) {
                    let (msg, sugg) = match direction {
                        Direction::From => ("convert from native endian", "from_ne_bytes"),
                        Direction::To => ("convert to native endian", "to_ne_bytes"),
                    };
                    diag.span_suggestion(sp, msg, sugg, Applicability::MaybeIncorrect);
                }
                if !ptr::addr_eq(lint, LITTLE_ENDIAN_BYTES) && is_lint_allowed(cx, LITTLE_ENDIAN_BYTES, e.hir_id) {
                    let (msg, sugg) = match direction {
                        Direction::From => ("convert from little endian", "from_le_bytes"),
                        Direction::To => ("convert to little endian", "to_le_bytes"),
                    };
                    diag.span_suggestion(sp, msg, sugg, Applicability::MaybeIncorrect);
                }
                if !ptr::addr_eq(lint, BIG_ENDIAN_BYTES) && is_lint_allowed(cx, BIG_ENDIAN_BYTES, e.hir_id) {
                    let (msg, sugg) = match direction {
                        Direction::From => ("convert from big endian", "from_be_bytes"),
                        Direction::To => ("convert to big endian", "to_be_bytes"),
                    };
                    diag.span_suggestion(sp, msg, sugg, Applicability::MaybeIncorrect);
                }
            });
        }
    }
}
