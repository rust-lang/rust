use std::ops::AddAssign;

use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::fn_has_unsatisfiable_preds;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
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
    /// ```rust
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
    /// ```rust
    /// struct NotSoLargeType(Box<[u8]>);
    ///
    /// fn foo() {
    ///     let _x1 = NotSoLargeType(vec![0; 500_000].into_boxed_slice());
    /// //                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Now heap allocated.
    /// //                                                                The size of `NotSoLargeType` is 16 bytes.
    /// //  ...
    /// }
    /// ```
    #[clippy::version = "1.71.0"]
    pub LARGE_STACK_FRAMES,
    nursery,
    "checks for functions that allocate a lot of stack space"
}

pub struct LargeStackFrames {
    maximum_allowed_size: u64,
}

impl LargeStackFrames {
    #[must_use]
    pub fn new(size: u64) -> Self {
        Self {
            maximum_allowed_size: size,
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

impl AddAssign<u64> for Space {
    fn add_assign(&mut self, rhs: u64) {
        if let Self::Used(lhs) = self {
            match lhs.checked_add(rhs) {
                Some(sum) => *self = Self::Used(sum),
                None => *self = Self::Overflow,
            }
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for LargeStackFrames {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: FnKind<'tcx>,
        _: &'tcx FnDecl<'tcx>,
        _: &'tcx Body<'tcx>,
        span: Span,
        local_def_id: LocalDefId,
    ) {
        let def_id = local_def_id.to_def_id();
        // Building MIR for `fn`s with unsatisfiable preds results in ICE.
        if fn_has_unsatisfiable_preds(cx, def_id) {
            return;
        }

        let mir = cx.tcx.optimized_mir(def_id);
        let param_env = cx.tcx.param_env(def_id);

        let mut frame_size = Space::Used(0);

        for local in &mir.local_decls {
            if let Ok(layout) = cx.tcx.layout_of(param_env.and(local.ty)) {
                frame_size += layout.size.bytes();
            }
        }

        if frame_size.exceeds_limit(self.maximum_allowed_size) {
            span_lint_and_note(
                cx,
                LARGE_STACK_FRAMES,
                span,
                "this function allocates a large amount of stack space",
                None,
                "allocating large amounts of stack space can overflow the stack",
            );
        }
    }
}
