use rustc_hir::{BinOpKind, Body, Expr, ExprKind, ImplItemKind, ItemKind, Node, TraitFn, TraitItem, TraitItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

mod cmp_null;
mod mut_from_ref;
mod ptr_arg;
mod ptr_eq;

declare_clippy_lint! {
    /// ### What it does
    /// This lint checks for function arguments of type `&String`, `&Vec`,
    /// `&PathBuf`, and `Cow<_>`. It will also suggest you replace `.clone()` calls
    /// with the appropriate `.to_owned()`/`to_string()` calls.
    ///
    /// ### Why is this bad?
    /// Requiring the argument to be of the specific type
    /// makes the function less useful for no benefit; slices in the form of `&[T]`
    /// or `&str` usually suffice and can be obtained from other types, too.
    ///
    /// ### Known problems
    /// There may be `fn(&Vec)`-typed references pointing to your function.
    /// If you have them, you will get a compiler error after applying this lint's
    /// suggestions. You then have the choice to undo your changes or change the
    /// type of the reference.
    ///
    /// Note that if the function is part of your public interface, there may be
    /// other crates referencing it, of which you may not be aware. Carefully
    /// deprecate the function before applying the lint suggestions in this case.
    ///
    /// ### Example
    /// ```ignore
    /// fn foo(&Vec<u32>) { .. }
    /// ```
    ///
    /// Use instead:
    /// ```ignore
    /// fn foo(&[u32]) { .. }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub PTR_ARG,
    style,
    "fn arguments of the type `&Vec<...>` or `&String`, suggesting to use `&[...]` or `&str` instead, respectively"
}

declare_clippy_lint! {
    /// ### What it does
    /// This lint checks for equality comparisons with `ptr::null` or `ptr::null_mut`
    ///
    /// ### Why is this bad?
    /// It's easier and more readable to use the inherent
    /// `.is_null()`
    /// method instead
    ///
    /// ### Example
    /// ```rust,ignore
    /// use std::ptr;
    ///
    /// if x == ptr::null() {
    ///     // ..
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// if x.is_null() {
    ///     // ..
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CMP_NULL,
    style,
    "comparing a pointer to a null pointer, suggesting to use `.is_null()` instead"
}

declare_clippy_lint! {
    /// ### What it does
    /// This lint checks for functions that take immutable references and return
    /// mutable ones. This will not trigger if no unsafe code exists as there
    /// are multiple safe functions which will do this transformation
    ///
    /// To be on the conservative side, if there's at least one mutable
    /// reference with the output lifetime, this lint will not trigger.
    ///
    /// ### Why is this bad?
    /// Creating a mutable reference which can be repeatably derived from an
    /// immutable reference is unsound as it allows creating multiple live
    /// mutable references to the same object.
    ///
    /// This [error](https://github.com/rust-lang/rust/issues/39465) actually
    /// lead to an interim Rust release 1.15.1.
    ///
    /// ### Known problems
    /// This pattern is used by memory allocators to allow allocating multiple
    /// objects while returning mutable references to each one. So long as
    /// different mutable references are returned each time such a function may
    /// be safe.
    ///
    /// ### Example
    /// ```ignore
    /// fn foo(&Foo) -> &mut Bar { .. }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MUT_FROM_REF,
    correctness,
    "fns that create mutable refs from immutable ref args"
}

declare_clippy_lint! {
    /// ### What it does
    /// Use `std::ptr::eq` when applicable
    ///
    /// ### Why is this bad?
    /// `ptr::eq` can be used to compare `&T` references
    /// (which coerce to `*const T` implicitly) by their address rather than
    /// comparing the values they point to.
    ///
    /// ### Example
    /// ```no_run
    /// let a = &[1, 2, 3];
    /// let b = &[1, 2, 3];
    ///
    /// assert!(a as *const _ as usize == b as *const _ as usize);
    /// ```
    /// Use instead:
    /// ```no_run
    /// let a = &[1, 2, 3];
    /// let b = &[1, 2, 3];
    ///
    /// assert!(std::ptr::eq(a, b));
    /// ```
    #[clippy::version = "1.49.0"]
    pub PTR_EQ,
    style,
    "use `std::ptr::eq` when comparing raw pointers"
}

declare_lint_pass!(Ptr => [PTR_ARG, CMP_NULL, MUT_FROM_REF, PTR_EQ]);

impl<'tcx> LateLintPass<'tcx> for Ptr {
    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Fn(sig, trait_method) = &item.kind {
            if matches!(trait_method, TraitFn::Provided(_)) {
                // Handled by `check_body`.
                return;
            }

            mut_from_ref::check(cx, sig, None);
            ptr_arg::check_trait_item(cx, item.owner_id, sig);
        }
    }

    fn check_body(&mut self, cx: &LateContext<'tcx>, body: &Body<'tcx>) {
        let mut parents = cx.tcx.hir_parent_iter(body.value.hir_id);
        let (item_id, sig, is_trait_item) = match parents.next() {
            Some((_, Node::Item(i))) => {
                if let ItemKind::Fn { sig, .. } = &i.kind {
                    (i.owner_id, sig, false)
                } else {
                    return;
                }
            },
            Some((_, Node::ImplItem(i))) => {
                if !matches!(parents.next(),
                    Some((_, Node::Item(i))) if matches!(&i.kind, ItemKind::Impl(i) if i.of_trait.is_none())
                ) {
                    return;
                }
                if let ImplItemKind::Fn(sig, _) = &i.kind {
                    (i.owner_id, sig, false)
                } else {
                    return;
                }
            },
            Some((_, Node::TraitItem(i))) => {
                if let TraitItemKind::Fn(sig, _) = &i.kind {
                    (i.owner_id, sig, true)
                } else {
                    return;
                }
            },
            _ => return,
        };

        mut_from_ref::check(cx, sig, Some(body));
        ptr_arg::check_body(cx, body, item_id, sig, is_trait_item);
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Binary(op, l, r) = expr.kind
            && (op.node == BinOpKind::Eq || op.node == BinOpKind::Ne)
        {
            #[expect(
                clippy::collapsible_if,
                reason = "the outer `if`s check the HIR, the inner ones run lints"
            )]
            if !cmp_null::check(cx, expr, op.node, l, r) {
                ptr_eq::check(cx, op.node, l, r, expr.span);
            }
        }
    }
}
