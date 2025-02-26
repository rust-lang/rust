use std::{fmt, ops};

use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::fn_has_unsatisfiable_preds;
use clippy_utils::source::SpanRangeExt;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl};
use rustc_lexer::is_ident;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions that use a lot of stack space.
    ///
    /// This often happens when constructing a large type, such as an array with a lot of elements,
    /// or constructing *many* smaller-but-still-large structs, or copying around a lot of large types.
    ///
    /// This lint is a more general version of [`large_stack_arrays`](https://rust-lang.github.io/rust-clippy/master/#large_stack_arrays)
    /// that is intended to look at functions as a whole instead of only individual array expressions inside of a function.
    ///
    /// ### Why is this bad?
    /// The stack region of memory is very limited in size (usually *much* smaller than the heap) and attempting to
    /// use too much will result in a stack overflow and crash the program.
    /// To avoid this, you should consider allocating large types on the heap instead (e.g. by boxing them).
    ///
    /// Keep in mind that the code path to construction of large types does not even need to be reachable;
    /// it purely needs to *exist* inside of the function to contribute to the stack size.
    /// For example, this causes a stack overflow even though the branch is unreachable:
    /// ```rust,ignore
    /// fn main() {
    ///     if false {
    ///         let x = [0u8; 10000000]; // 10 MB stack array
    ///         black_box(&x);
    ///     }
    /// }
    /// ```
    ///
    /// ### Known issues
    /// False positives. The stack size that clippy sees is an estimated value and can be vastly different
    /// from the actual stack usage after optimizations passes have run (especially true in release mode).
    /// Modern compilers are very smart and are able to optimize away a lot of unnecessary stack allocations.
    /// In debug mode however, it is usually more accurate.
    ///
    /// This lint works by summing up the size of all variables that the user typed, variables that were
    /// implicitly introduced by the compiler for temporaries, function arguments and the return value,
    /// and comparing them against a (configurable, but high-by-default).
    ///
    /// ### Example
    /// This function creates four 500 KB arrays on the stack. Quite big but just small enough to not trigger `large_stack_arrays`.
    /// However, looking at the function as a whole, it's clear that this uses a lot of stack space.
    /// ```no_run
    /// struct QuiteLargeType([u8; 500_000]);
    /// fn foo() {
    ///     // ... some function that uses a lot of stack space ...
    ///     let _x1 = QuiteLargeType([0; 500_000]);
    ///     let _x2 = QuiteLargeType([0; 500_000]);
    ///     let _x3 = QuiteLargeType([0; 500_000]);
    ///     let _x4 = QuiteLargeType([0; 500_000]);
    /// }
    /// ```
    ///
    /// Instead of doing this, allocate the arrays on the heap.
    /// This currently requires going through a `Vec` first and then converting it to a `Box`:
    /// ```no_run
    /// struct NotSoLargeType(Box<[u8]>);
    ///
    /// fn foo() {
    ///     let _x1 = NotSoLargeType(vec![0; 500_000].into_boxed_slice());
    /// //                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Now heap allocated.
    /// //                                                                The size of `NotSoLargeType` is 16 bytes.
    /// //  ...
    /// }
    /// ```
    #[clippy::version = "1.72.0"]
    pub LARGE_STACK_FRAMES,
    nursery,
    "checks for functions that allocate a lot of stack space"
}

pub struct LargeStackFrames {
    maximum_allowed_size: u64,
}

impl LargeStackFrames {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            maximum_allowed_size: conf.stack_size_threshold,
        }
    }
}

impl_lint_pass!(LargeStackFrames => [LARGE_STACK_FRAMES]);

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

impl<'tcx> LateLintPass<'tcx> for LargeStackFrames {
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
        if fn_has_unsatisfiable_preds(cx, def_id) {
            return;
        }

        let mir = cx.tcx.optimized_mir(def_id);
        let typing_env = mir.typing_env(cx.tcx);

        let sizes_of_locals = || {
            mir.local_decls.iter().filter_map(|local| {
                let layout = cx.tcx.layout_of(typing_env.as_query_input(local.ty)).ok()?;
                Some((local, layout.size.bytes()))
            })
        };

        let frame_size = sizes_of_locals().fold(Space::Used(0), |sum, (_, size)| sum + size);

        let limit = self.maximum_allowed_size;
        if frame_size.exceeds_limit(limit) {
            // Point at just the function name if possible, because lints that span
            // the entire body and don't have to are less legible.
            let fn_span = match fn_kind {
                FnKind::ItemFn(ident, _, _) | FnKind::Method(ident, _) => ident.span,
                FnKind::Closure => entire_fn_span,
            };

            span_lint_and_then(
                cx,
                LARGE_STACK_FRAMES,
                fn_span,
                format!("this function may allocate {frame_size} on the stack"),
                |diag| {
                    // Point out the largest individual contribution to this size, because
                    // it is the most likely to be unintentionally large.
                    if let Some((local, size)) = sizes_of_locals().max_by_key(|&(_, size)| size) {
                        let local_span: Span = local.source_info.span;
                        let size = Space::Used(size); // pluralizes for us
                        let ty = local.ty;

                        // TODO: Is there a cleaner, robust way to ask this question?
                        // The obvious `LocalDecl::is_user_variable()` panics on "unwrapping cross-crate data",
                        // and that doesn't get us the true name in scope rather than the span text either.
                        if let Some(name) = local_span.get_source_text(cx)
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
                },
            );
        }
    }
}
