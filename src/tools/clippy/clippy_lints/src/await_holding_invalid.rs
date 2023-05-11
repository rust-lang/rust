use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{match_def_path, paths};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_hir::{AsyncGeneratorKind, Body, BodyId, GeneratorKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::GeneratorInteriorTypeCause;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, Span};

use crate::utils::conf::DisallowedPath;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to await while holding a non-async-aware MutexGuard.
    ///
    /// ### Why is this bad?
    /// The Mutex types found in std::sync and parking_lot
    /// are not designed to operate in an async context across await points.
    ///
    /// There are two potential solutions. One is to use an async-aware Mutex
    /// type. Many asynchronous foundation crates provide such a Mutex type. The
    /// other solution is to ensure the mutex is unlocked before calling await,
    /// either by introducing a scope or an explicit call to Drop::drop.
    ///
    /// ### Known problems
    /// Will report false positive for explicitly dropped guards
    /// ([#6446](https://github.com/rust-lang/rust-clippy/issues/6446)). A workaround for this is
    /// to wrap the `.lock()` call in a block instead of explicitly dropping the guard.
    ///
    /// ### Example
    /// ```rust
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
    /// ```rust
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
    /// Checks for calls to await while holding a `RefCell` `Ref` or `RefMut`.
    ///
    /// ### Why is this bad?
    /// `RefCell` refs only check for exclusive mutable access
    /// at runtime. Holding onto a `RefCell` ref across an `await` suspension point
    /// risks panics from a mutable ref shared while other refs are outstanding.
    ///
    /// ### Known problems
    /// Will report false positive for explicitly dropped refs
    /// ([#6353](https://github.com/rust-lang/rust-clippy/issues/6353)). A workaround for this is
    /// to wrap the `.borrow[_mut]()` call in a block instead of explicitly dropping the ref.
    ///
    /// ### Example
    /// ```rust
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
    /// ```rust
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
    /// Allows users to configure types which should not be held across `await`
    /// suspension points.
    ///
    /// ### Why is this bad?
    /// There are some types which are perfectly "safe" to be used concurrently
    /// from a memory access perspective but will cause bugs at runtime if they
    /// are held in such a way.
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
    /// ```rust
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

#[derive(Debug)]
pub struct AwaitHolding {
    conf_invalid_types: Vec<DisallowedPath>,
    def_ids: FxHashMap<DefId, DisallowedPath>,
}

impl AwaitHolding {
    pub(crate) fn new(conf_invalid_types: Vec<DisallowedPath>) -> Self {
        Self {
            conf_invalid_types,
            def_ids: FxHashMap::default(),
        }
    }
}

impl LateLintPass<'_> for AwaitHolding {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        for conf in &self.conf_invalid_types {
            let segs: Vec<_> = conf.path().split("::").collect();
            for id in clippy_utils::def_path_def_ids(cx, &segs) {
                self.def_ids.insert(id, conf.clone());
            }
        }
    }

    fn check_body(&mut self, cx: &LateContext<'_>, body: &'_ Body<'_>) {
        use AsyncGeneratorKind::{Block, Closure, Fn};
        if let Some(GeneratorKind::Async(Block | Closure | Fn)) = body.generator_kind {
            let body_id = BodyId {
                hir_id: body.value.hir_id,
            };
            let typeck_results = cx.tcx.typeck_body(body_id);
            self.check_interior_types(
                cx,
                typeck_results.generator_interior_types.as_ref().skip_binder(),
                body.value.span,
            );
        }
    }
}

impl AwaitHolding {
    fn check_interior_types(&self, cx: &LateContext<'_>, ty_causes: &[GeneratorInteriorTypeCause<'_>], span: Span) {
        for ty_cause in ty_causes {
            if let rustc_middle::ty::Adt(adt, _) = ty_cause.ty.kind() {
                if is_mutex_guard(cx, adt.did()) {
                    span_lint_and_then(
                        cx,
                        AWAIT_HOLDING_LOCK,
                        ty_cause.span,
                        "this `MutexGuard` is held across an `await` point",
                        |diag| {
                            diag.help(
                                "consider using an async-aware `Mutex` type or ensuring the \
                                `MutexGuard` is dropped before calling await",
                            );
                            diag.span_note(
                                ty_cause.scope_span.unwrap_or(span),
                                "these are all the `await` points this lock is held through",
                            );
                        },
                    );
                } else if is_refcell_ref(cx, adt.did()) {
                    span_lint_and_then(
                        cx,
                        AWAIT_HOLDING_REFCELL_REF,
                        ty_cause.span,
                        "this `RefCell` reference is held across an `await` point",
                        |diag| {
                            diag.help("ensure the reference is dropped before calling `await`");
                            diag.span_note(
                                ty_cause.scope_span.unwrap_or(span),
                                "these are all the `await` points this reference is held through",
                            );
                        },
                    );
                } else if let Some(disallowed) = self.def_ids.get(&adt.did()) {
                    emit_invalid_type(cx, ty_cause.span, disallowed);
                }
            }
        }
    }
}

fn emit_invalid_type(cx: &LateContext<'_>, span: Span, disallowed: &DisallowedPath) {
    span_lint_and_then(
        cx,
        AWAIT_HOLDING_INVALID_TYPE,
        span,
        &format!(
            "`{}` may not be held across an `await` point per `clippy.toml`",
            disallowed.path()
        ),
        |diag| {
            if let Some(reason) = disallowed.reason() {
                diag.note(reason);
            }
        },
    );
}

fn is_mutex_guard(cx: &LateContext<'_>, def_id: DefId) -> bool {
    cx.tcx.is_diagnostic_item(sym::MutexGuard, def_id)
        || cx.tcx.is_diagnostic_item(sym::RwLockReadGuard, def_id)
        || cx.tcx.is_diagnostic_item(sym::RwLockWriteGuard, def_id)
        || match_def_path(cx, def_id, &paths::PARKING_LOT_MUTEX_GUARD)
        || match_def_path(cx, def_id, &paths::PARKING_LOT_RWLOCK_READ_GUARD)
        || match_def_path(cx, def_id, &paths::PARKING_LOT_RWLOCK_WRITE_GUARD)
}

fn is_refcell_ref(cx: &LateContext<'_>, def_id: DefId) -> bool {
    match_def_path(cx, def_id, &paths::REFCELL_REF) || match_def_path(cx, def_id, &paths::REFCELL_REFMUT)
}
