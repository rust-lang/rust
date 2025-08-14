use clippy_config::Conf;
use clippy_config::types::{DisallowedPathWithoutReplacement, create_disallowed_map};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::paths::{self, PathNS};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, DefIdMap};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::CoroutineLayout;
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `await` while holding a non-async-aware
    /// `MutexGuard`.
    ///
    /// ### Why is this bad?
    /// The Mutex types found in [`std::sync`](https://doc.rust-lang.org/stable/std/sync/) and
    /// [`parking_lot`](https://docs.rs/parking_lot/latest/parking_lot/) are
    /// not designed to operate in an async context across await points.
    ///
    /// There are two potential solutions. One is to use an async-aware `Mutex`
    /// type. Many asynchronous foundation crates provide such a `Mutex` type.
    /// The other solution is to ensure the mutex is unlocked before calling
    /// `await`, either by introducing a scope or an explicit call to
    /// [`Drop::drop`](https://doc.rust-lang.org/std/ops/trait.Drop.html).
    ///
    /// ### Known problems
    /// Will report false positive for explicitly dropped guards
    /// ([#6446](https://github.com/rust-lang/rust-clippy/issues/6446)). A
    /// workaround for this is to wrap the `.lock()` call in a block instead of
    /// explicitly dropping the guard.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::sync::Mutex;
    /// # async fn baz() {}
    /// async fn foo(x: &Mutex<u32>) {
    ///   let mut guard = x.lock().unwrap();
    ///   *guard += 1;
    ///   baz().await;
    /// }
    ///
    /// async fn bar(x: &Mutex<u32>) {
    ///   let mut guard = x.lock().unwrap();
    ///   *guard += 1;
    ///   drop(guard); // explicit drop
    ///   baz().await;
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # use std::sync::Mutex;
    /// # async fn baz() {}
    /// async fn foo(x: &Mutex<u32>) {
    ///   {
    ///     let mut guard = x.lock().unwrap();
    ///     *guard += 1;
    ///   }
    ///   baz().await;
    /// }
    ///
    /// async fn bar(x: &Mutex<u32>) {
    ///   {
    ///     let mut guard = x.lock().unwrap();
    ///     *guard += 1;
    ///   } // guard dropped here at end of scope
    ///   baz().await;
    /// }
    /// ```
    #[clippy::version = "1.45.0"]
    pub AWAIT_HOLDING_LOCK,
    suspicious,
    "inside an async function, holding a `MutexGuard` while calling `await`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `await` while holding a `RefCell`, `Ref`, or `RefMut`.
    ///
    /// ### Why is this bad?
    /// `RefCell` refs only check for exclusive mutable access
    /// at runtime. Holding a `RefCell` ref across an await suspension point
    /// risks panics from a mutable ref shared while other refs are outstanding.
    ///
    /// ### Known problems
    /// Will report false positive for explicitly dropped refs
    /// ([#6353](https://github.com/rust-lang/rust-clippy/issues/6353)). A workaround for this is
    /// to wrap the `.borrow[_mut]()` call in a block instead of explicitly dropping the ref.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::cell::RefCell;
    /// # async fn baz() {}
    /// async fn foo(x: &RefCell<u32>) {
    ///   let mut y = x.borrow_mut();
    ///   *y += 1;
    ///   baz().await;
    /// }
    ///
    /// async fn bar(x: &RefCell<u32>) {
    ///   let mut y = x.borrow_mut();
    ///   *y += 1;
    ///   drop(y); // explicit drop
    ///   baz().await;
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # use std::cell::RefCell;
    /// # async fn baz() {}
    /// async fn foo(x: &RefCell<u32>) {
    ///   {
    ///      let mut y = x.borrow_mut();
    ///      *y += 1;
    ///   }
    ///   baz().await;
    /// }
    ///
    /// async fn bar(x: &RefCell<u32>) {
    ///   {
    ///     let mut y = x.borrow_mut();
    ///     *y += 1;
    ///   } // y dropped here at end of scope
    ///   baz().await;
    /// }
    /// ```
    #[clippy::version = "1.49.0"]
    pub AWAIT_HOLDING_REFCELL_REF,
    suspicious,
    "inside an async function, holding a `RefCell` ref while calling `await`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Allows users to configure types which should not be held across await
    /// suspension points.
    ///
    /// ### Why is this bad?
    /// There are some types which are perfectly safe to use concurrently from
    /// a memory access perspective, but that will cause bugs at runtime if
    /// they are held in such a way.
    ///
    /// ### Example
    ///
    /// ```toml
    /// await-holding-invalid-types = [
    ///   # You can specify a type name
    ///   "CustomLockType",
    ///   # You can (optionally) specify a reason
    ///   { path = "OtherCustomLockType", reason = "Relies on a thread local" }
    /// ]
    /// ```
    ///
    /// ```no_run
    /// # async fn baz() {}
    /// struct CustomLockType;
    /// struct OtherCustomLockType;
    /// async fn foo() {
    ///   let _x = CustomLockType;
    ///   let _y = OtherCustomLockType;
    ///   baz().await; // Lint violation
    /// }
    /// ```
    #[clippy::version = "1.62.0"]
    pub AWAIT_HOLDING_INVALID_TYPE,
    suspicious,
    "holding a type across an await point which is not allowed to be held as per the configuration"
}

impl_lint_pass!(AwaitHolding => [AWAIT_HOLDING_LOCK, AWAIT_HOLDING_REFCELL_REF, AWAIT_HOLDING_INVALID_TYPE]);

pub struct AwaitHolding {
    def_ids: DefIdMap<(&'static str, &'static DisallowedPathWithoutReplacement)>,
}

impl AwaitHolding {
    pub(crate) fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        let (def_ids, _) = create_disallowed_map(
            tcx,
            &conf.await_holding_invalid_types,
            PathNS::Type,
            crate::disallowed_types::def_kind_predicate,
            "type",
            false,
        );
        Self { def_ids }
    }
}

impl<'tcx> LateLintPass<'tcx> for AwaitHolding {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Closure(hir::Closure {
            kind: hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _)),
            def_id,
            ..
        }) = expr.kind
            && let Some(coroutine_layout) = cx.tcx.mir_coroutine_witnesses(*def_id)
        {
            self.check_interior_types(cx, coroutine_layout);
        }
    }
}

impl AwaitHolding {
    fn check_interior_types(&self, cx: &LateContext<'_>, coroutine: &CoroutineLayout<'_>) {
        for (ty_index, ty_cause) in coroutine.field_tys.iter_enumerated() {
            if let rustc_middle::ty::Adt(adt, _) = ty_cause.ty.kind() {
                let await_points = || {
                    coroutine
                        .variant_source_info
                        .iter_enumerated()
                        .filter_map(|(variant, source_info)| {
                            coroutine.variant_fields[variant]
                                .raw
                                .contains(&ty_index)
                                .then_some(source_info.span)
                        })
                        .collect::<Vec<_>>()
                };
                if is_mutex_guard(cx, adt.did()) {
                    span_lint_and_then(
                        cx,
                        AWAIT_HOLDING_LOCK,
                        ty_cause.source_info.span,
                        "this `MutexGuard` is held across an await point",
                        |diag| {
                            diag.help(
                                "consider using an async-aware `Mutex` type or ensuring the \
                                `MutexGuard` is dropped before calling `await`",
                            );
                            diag.span_note(
                                await_points(),
                                "these are all the await points this lock is held through",
                            );
                        },
                    );
                } else if is_refcell_ref(cx, adt.did()) {
                    span_lint_and_then(
                        cx,
                        AWAIT_HOLDING_REFCELL_REF,
                        ty_cause.source_info.span,
                        "this `RefCell` reference is held across an await point",
                        |diag| {
                            diag.help("ensure the reference is dropped before calling `await`");
                            diag.span_note(
                                await_points(),
                                "these are all the await points this reference is held through",
                            );
                        },
                    );
                } else if let Some(&(path, disallowed_path)) = self.def_ids.get(&adt.did()) {
                    emit_invalid_type(cx, ty_cause.source_info.span, path, disallowed_path);
                }
            }
        }
    }
}

fn emit_invalid_type(
    cx: &LateContext<'_>,
    span: Span,
    path: &'static str,
    disallowed_path: &'static DisallowedPathWithoutReplacement,
) {
    span_lint_and_then(
        cx,
        AWAIT_HOLDING_INVALID_TYPE,
        span,
        format!("holding a disallowed type across an await point `{path}`"),
        disallowed_path.diag_amendment(span),
    );
}

fn is_mutex_guard(cx: &LateContext<'_>, def_id: DefId) -> bool {
    match cx.tcx.get_diagnostic_name(def_id) {
        Some(name) => matches!(name, sym::MutexGuard | sym::RwLockReadGuard | sym::RwLockWriteGuard),
        None => paths::PARKING_LOT_GUARDS.iter().any(|guard| guard.matches(cx, def_id)),
    }
}

fn is_refcell_ref(cx: &LateContext<'_>, def_id: DefId) -> bool {
    matches!(
        cx.tcx.get_diagnostic_name(def_id),
        Some(sym::RefCellRef | sym::RefCellRefMut)
    )
}
