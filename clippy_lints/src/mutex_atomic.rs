use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::MaybeDef;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::ty_from_hir_ty;
use rustc_errors::{Applicability, Diag};
use rustc_hir::{Expr, ExprKind, Item, ItemKind, LetStmt, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::Mutability;
use rustc_middle::ty::{self, IntTy, Ty, UintTy};
use rustc_session::declare_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `Mutex<X>` where an atomic will do.
    ///
    /// ### Why restrict this?
    /// Using a mutex just to make access to a plain bool or
    /// reference sequential is shooting flies with cannons.
    /// `std::sync::atomic::AtomicBool` and `std::sync::atomic::AtomicPtr` are leaner and
    /// faster.
    ///
    /// On the other hand, `Mutex`es are, in general, easier to
    /// verify correctness. An atomic does not behave the same as
    /// an equivalent mutex. See [this issue](https://github.com/rust-lang/rust-clippy/issues/4295)'s
    /// commentary for more details.
    ///
    /// ### Known problems
    /// * This lint cannot detect if the mutex is actually used
    ///   for waiting before a critical section.
    /// * This lint has a false positive that warns without considering the case
    ///   where `Mutex` is used together with `Condvar`.
    ///
    /// ### Example
    /// ```no_run
    /// # let y = true;
    /// # use std::sync::Mutex;
    /// let x = Mutex::new(&y);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let y = true;
    /// # use std::sync::atomic::AtomicBool;
    /// let x = AtomicBool::new(y);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MUTEX_ATOMIC,
    restriction,
    "using a mutex where an atomic value could be used instead."
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `Mutex<X>` where `X` is an integral
    /// type.
    ///
    /// ### Why restrict this?
    /// Using a mutex just to make access to a plain integer
    /// sequential is
    /// shooting flies with cannons. `std::sync::atomic::AtomicUsize` is leaner and faster.
    ///
    /// On the other hand, `Mutex`es are, in general, easier to
    /// verify correctness. An atomic does not behave the same as
    /// an equivalent mutex. See [this issue](https://github.com/rust-lang/rust-clippy/issues/4295)'s
    /// commentary for more details.
    ///
    /// ### Known problems
    /// * This lint cannot detect if the mutex is actually used
    ///   for waiting before a critical section.
    /// * This lint has a false positive that warns without considering the case
    ///   where `Mutex` is used together with `Condvar`.
    /// * This lint suggest using `AtomicU64` instead of `Mutex<u64>`, but
    ///   `AtomicU64` is not available on some 32-bit platforms.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::sync::Mutex;
    /// let x = Mutex::new(0usize);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # use std::sync::atomic::AtomicUsize;
    /// let x = AtomicUsize::new(0usize);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MUTEX_INTEGER,
    restriction,
    "using a mutex for an integer type"
}

declare_lint_pass!(Mutex => [MUTEX_ATOMIC, MUTEX_INTEGER]);

// NOTE: we don't use `check_expr` because that would make us lint every _use_ of such mutexes, not
// just their definitions
impl<'tcx> LateLintPass<'tcx> for Mutex {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if !item.span.from_expansion()
            && let ItemKind::Static(_, _, ty, body_id) = item.kind
        {
            let body = cx.tcx.hir_body(body_id);
            let mid_ty = ty_from_hir_ty(cx, ty);
            check_expr(cx, body.value.peel_blocks(), mid_ty);
        }
    }
    fn check_local(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx LetStmt<'_>) {
        if !stmt.span.from_expansion()
            && let Some(init) = stmt.init
        {
            let mid_ty = cx.typeck_results().expr_ty(init);
            check_expr(cx, init.peel_blocks(), mid_ty);
        }
    }
}

fn check_expr<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>, ty: Ty<'tcx>) {
    if let ty::Adt(_, subst) = ty.kind()
        && ty.is_diag_item(cx, sym::Mutex)
        && let mutex_param = subst.type_at(0)
        && let Some(atomic_name) = get_atomic_name(mutex_param)
    {
        let msg = "using a `Mutex` where an atomic would do";
        let diag = |diag: &mut Diag<'_, _>| {
            // if `expr = Mutex::new(arg)`, we can try emitting a suggestion
            if let ExprKind::Call(qpath, [arg]) = expr.kind
                && let ExprKind::Path(QPath::TypeRelative(_mutex, new)) = qpath.kind
                && new.ident.name == sym::new
            {
                let mut applicability = Applicability::MaybeIncorrect;
                let arg = Sugg::hir_with_applicability(cx, arg, "_", &mut applicability);

                let suggs = vec![(expr.span, format!("std::sync::atomic::{atomic_name}::new({arg})"))];
                diag.multipart_suggestion("try", suggs, applicability);
            } else {
                diag.help(format!("consider using an `{atomic_name}` instead"));
            }
            diag.help("if you just want the locking behavior and not the internal type, consider using `Mutex<()>`");
        };
        match *mutex_param.kind() {
            ty::Uint(t) if t != UintTy::Usize => span_lint_and_then(cx, MUTEX_INTEGER, expr.span, msg, diag),
            ty::Int(t) if t != IntTy::Isize => span_lint_and_then(cx, MUTEX_INTEGER, expr.span, msg, diag),
            _ => span_lint_and_then(cx, MUTEX_ATOMIC, expr.span, msg, diag),
        }
    }
}

fn get_atomic_name(ty: Ty<'_>) -> Option<&'static str> {
    match ty.kind() {
        ty::Bool => Some("AtomicBool"),
        ty::Uint(uint_ty) => {
            match uint_ty {
                UintTy::U8 => Some("AtomicU8"),
                UintTy::U16 => Some("AtomicU16"),
                UintTy::U32 => Some("AtomicU32"),
                UintTy::U64 => Some("AtomicU64"),
                UintTy::Usize => Some("AtomicUsize"),
                // `AtomicU128` is unstable and only available on a few platforms: https://github.com/rust-lang/rust/issues/99069
                UintTy::U128 => None,
            }
        },
        ty::Int(int_ty) => {
            match int_ty {
                IntTy::I8 => Some("AtomicI8"),
                IntTy::I16 => Some("AtomicI16"),
                IntTy::I32 => Some("AtomicI32"),
                IntTy::I64 => Some("AtomicI64"),
                IntTy::Isize => Some("AtomicIsize"),
                // `AtomicU128` is unstable and only available on a few platforms: https://github.com/rust-lang/rust/issues/99069
                IntTy::I128 => None,
            }
        },
        // `AtomicPtr` only accepts `*mut T`
        ty::RawPtr(_, Mutability::Mut) => Some("AtomicPtr"),
        _ => None,
    }
}
