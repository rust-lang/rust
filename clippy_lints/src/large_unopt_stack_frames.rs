use std::{fmt, ops};

use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::SpanExt as _;
use clippy_utils::{fn_has_unsatisfiable_clauses, is_entrypoint_fn, is_in_test};
use rustc_errors::Diag;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl};
use rustc_lexer::is_ident;
use rustc_lint::{LateContext, LateLintPass, impl_lint_pass};
use rustc_span::{Span, SyntaxContext};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions that use a lot of stack space when optimizations are disabled
    /// (e.g. debug builds).
    ///
    /// ### Why is this bad?
    /// The stack region of memory is normally very limited in size (measuring a few MBs) and any
    /// attempt to allocate beyond that will cause a stack overflow and crash the program. Due to
    /// how rustc creates a function's stack frame, it's easy to end up with surprisingly large
    /// stack frames when optimizations are disabled.
    ///
    /// With `opt-level` set to zero (the default for debug builds) rustc creates stack frames large
    /// enough to hold all named and temporary values simultaneously. Because this doesn't take the
    /// lifetimes of the individual values into account, stack frames will almost always be larger
    /// than needed. In more extreme cases they can be over a megabyte larger.
    ///
    /// ### Examples
    /// ```rust,ignore
    /// let a = Box::new(mk_large_value1());
    /// let b = Box::new(mk_large_value2());
    /// let c = Box::new(mk_large_value3());
    /// let d = Box::new(mk_large_value4());
    /// ```
    ///
    /// The unoptimized stack frame will be large enough to hold all four temporaries simultaneously
    /// even though this function only ever needs to hold at most one on the stack at any time. For
    /// box allocation in particular this can be fixed with a helper function. e.g.
    ///
    /// ```rust,ignore
    /// fn mk_box<T>(f: impl FnOnce() -> T) -> Box<T> {
    ///     Box::new(f())
    /// }
    ///
    /// let a = mk_box(|| mk_large_value1());
    /// let b = mk_box(|| mk_large_value2());
    /// let c = mk_box(|| mk_large_value3());
    /// let d = mk_box(|| mk_large_value4());
    /// ```
    ///
    /// This will split the single large frame into multiple smaller and shorter-lived frames. More
    /// generally any function can be split such that the large values are distributed amongst the
    /// sub functions.
    ///
    /// ---
    ///
    /// ```rust,ignore
    /// const X: LargeType = mk_large_value();
    ///
    /// foo(X.field1);
    /// foo(X.field2);
    /// foo(X.field3);
    /// foo(X.field4);
    /// ```
    ///
    /// Unoptimized builds will always materialize the entire constant every time it's referenced,
    /// even if only a small portion of it is needed. In this case the stack frame will be large
    /// enough to fit four instances simultaneously. When only a small subset of the constant is
    /// required, the access can be wrapped in a const block to force it to happen during constant
    /// evaluation. e.g.
    ///
    /// ```rust,ignore
    /// const X: LargeType = mk_large_value();
    ///
    /// foo(const { x.field1 });
    /// foo(const { x.field2 });
    /// foo(const { x.field3 });
    /// foo(const { x.field4 });
    /// ```
    ///
    /// This will cause the stack frame to only contain the four fields rather than four instances
    /// of the entire constant. Alternatively, the constant could be made a static which would also
    /// avoid the duplicate values on the stack.
    ///
    /// ---
    ///
    /// If the stack size ultimately cannot be sufficiently reduced and the program is crashing due
    /// to a stack overflow then minimal optimizations can be enabled. This will, however, result in
    /// a slightly degraded debug experience and longer build times. This can be done with cargo by
    /// setting the appropriate field in `cargo.toml`. e.g.
    ///
    /// ```toml
    /// [profile.dev]
    /// opt-level = 1
    /// ```
    ///
    /// When doing so you may wish to disable this lint as the size calculated will no longer
    /// represent the actual size of a function's stack frame.
    #[clippy::version = "1.72.0"]
    pub LARGE_UNOPT_STACK_FRAMES,
    nursery,
    "checks for functions that allocate a lot of stack space"
}

impl_lint_pass!(LargeUnoptStackFrames => [LARGE_UNOPT_STACK_FRAMES]);

pub struct LargeUnoptStackFrames {
    maximum_allowed_size: u64,
    allow_large_stack_frames_in_tests: bool,
}

impl LargeUnoptStackFrames {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            maximum_allowed_size: conf.stack_size_threshold,
            allow_large_stack_frames_in_tests: conf.allow_large_stack_frames_in_tests,
        }
    }
}

#[derive(Copy, Clone)]
enum Space {
    Used(u64),
    Overflow,
}

impl Space {
    pub fn exceeds_limit(self, limit: u64) -> bool {
        match self {
            Self::Used(used) => used > limit,
            Self::Overflow => true,
        }
    }
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Space::Used(1) => write!(f, "1 byte"),
            Space::Used(n) => write!(f, "{n} bytes"),
            Space::Overflow => write!(f, "over 2⁶⁴-1 bytes"),
        }
    }
}

impl ops::Add<u64> for Space {
    type Output = Self;
    fn add(self, rhs: u64) -> Self {
        match self {
            Self::Used(lhs) => match lhs.checked_add(rhs) {
                Some(sum) => Self::Used(sum),
                None => Self::Overflow,
            },
            Self::Overflow => self,
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for LargeUnoptStackFrames {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'tcx>,
        _: &'tcx FnDecl<'tcx>,
        _: &'tcx Body<'tcx>,
        entire_fn_span: Span,
        local_def_id: LocalDefId,
    ) {
        let def_id = local_def_id.to_def_id();
        // Building MIR for `fn`s with unsatisfiable preds results in ICE.
        if fn_has_unsatisfiable_clauses(cx, def_id) {
            return;
        }

        let mir = cx.tcx.optimized_mir(def_id);
        let typing_env = mir.typing_env(cx.tcx);

        let sizes_of_locals = mir
            .local_decls
            .iter()
            .filter_map(|local| {
                let layout = cx.tcx.layout_of(typing_env.as_query_input(local.ty)).ok()?;
                Some((local, layout.size.bytes()))
            })
            .collect::<Vec<_>>();

        let frame_size = sizes_of_locals
            .iter()
            .fold(Space::Used(0), |sum, (_, size)| sum + *size);

        let limit = self.maximum_allowed_size;
        if frame_size.exceeds_limit(limit) {
            // Point at just the function name if possible, because lints that span
            // the entire body and don't have to are less legible.
            let (fn_span, fn_name) = match fn_kind {
                FnKind::ItemFn(ident, _, _) => (ident.span, format!("function `{}`", ident.name)),
                FnKind::Method(ident, _) => (ident.span, format!("method `{}`", ident.name)),
                FnKind::Closure => (entire_fn_span, "closure".to_string()),
            };

            // Don't lint inside tests if configured to not do so.
            if self.allow_large_stack_frames_in_tests && is_in_test(cx.tcx, cx.tcx.local_def_id_to_hir_id(local_def_id))
            {
                return;
            }

            let explain_lint = |diag: &mut Diag<'_>, ctxt: SyntaxContext| {
                // Point out the largest individual contribution to this size, because
                // it is the most likely to be unintentionally large.
                if let Some((local, size)) = sizes_of_locals.iter().max_by_key(|&(_, size)| size)
                    && let local_span = local.source_info.span
                    && local_span.ctxt() == ctxt
                {
                    let size = Space::Used(*size); // pluralizes for us
                    let ty = local.ty;

                    // TODO: Is there a cleaner, robust way to ask this question?
                    // The obvious `LocalDecl::is_user_variable()` panics on "unwrapping cross-crate data",
                    // and that doesn't get us the true name in scope rather than the span text either.
                    if let Some(name) = local_span.get_text(cx)
                        && is_ident(&name)
                    {
                        // If the local is an ordinary named variable,
                        // print its name rather than relying solely on the span.
                        diag.span_label(
                            local_span,
                            format!("`{name}` is the largest part, at {size} for type `{ty}`"),
                        );
                    } else {
                        diag.span_label(
                            local_span,
                            format!("this is the largest part, at {size} for type `{ty}`"),
                        );
                    }
                }

                // Explain why we are linting this and not other functions.
                diag.note(format!(
                    "{frame_size} is larger than Clippy's configured `stack-size-threshold` of {limit}"
                ));

                // Explain why the user should care, briefly.
                diag.note_once(
                    "allocating large amounts of stack space can overflow the stack \
                        and cause the program to abort",
                );
            };

            if fn_span.from_expansion() {
                // Don't lint on the main function generated by `--test` target
                if cx.tcx.sess.is_test_crate() && is_entrypoint_fn(cx, local_def_id.to_def_id()) {
                    return;
                }

                let is_from_external_macro = fn_span.in_external_macro(cx.tcx.sess.source_map());
                span_lint_and_then(
                    cx,
                    LARGE_UNOPT_STACK_FRAMES,
                    fn_span.source_callsite(),
                    format!(
                        "{} generated by this macro may allocate a lot of stack space",
                        if is_from_external_macro {
                            cx.tcx.def_descr(local_def_id.into())
                        } else {
                            fn_name.as_str()
                        }
                    ),
                    |diag| {
                        if is_from_external_macro {
                            return;
                        }

                        diag.span_label(
                            fn_span,
                            format!(
                                "this {} has a stack frame size of {frame_size}",
                                cx.tcx.def_descr(local_def_id.into())
                            ),
                        );

                        explain_lint(diag, fn_span.ctxt());
                    },
                );
                return;
            }

            span_lint_and_then(
                cx,
                LARGE_UNOPT_STACK_FRAMES,
                fn_span,
                format!("this function may allocate {frame_size} on the stack"),
                |diag| {
                    explain_lint(diag, SyntaxContext::root());
                },
            );
        }
    }
}
