use rustc_ast::visit::{visit_opt, walk_list};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::Res;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{FnKind, Visitor, walk_expr};
use rustc_hir::{Block, Body, Expr, ExprKind, FnDecl, FnRetTy, LangItem, TyKind, find_attr};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::{Span, sym};

use crate::lints::{DanglingPointersFromLocals, DanglingPointersFromTemporaries};
use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `dangling_pointers_from_temporaries` lint detects getting a pointer to data
    /// of a temporary that will immediately get dropped.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// # unsafe fn use_data(ptr: *const u8) { }
    /// fn gather_and_use(bytes: impl Iterator<Item = u8>) {
    ///     let x: *const u8 = bytes.collect::<Vec<u8>>().as_ptr();
    ///     unsafe { use_data(x) }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Getting a pointer from a temporary value will not prolong its lifetime,
    /// which means that the value can be dropped and the allocation freed
    /// while the pointer still exists, making the pointer dangling.
    /// This is not an error (as far as the type system is concerned)
    /// but probably is not what the user intended either.
    ///
    /// If you need stronger guarantees, consider using references instead,
    /// as they are statically verified by the borrow-checker to never dangle.
    pub DANGLING_POINTERS_FROM_TEMPORARIES,
    Warn,
    "detects getting a pointer from a temporary"
}

declare_lint! {
    /// The `dangling_pointers_from_locals` lint detects getting a pointer to data
    /// of a local that will be dropped at the end of the function.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn f() -> *const u8 {
    ///     let x = 0;
    ///     &x // returns a dangling ptr to `x`
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Returning a pointer from a local value will not prolong its lifetime,
    /// which means that the value can be dropped and the allocation freed
    /// while the pointer still exists, making the pointer dangling.
    /// This is not an error (as far as the type system is concerned)
    /// but probably is not what the user intended either.
    ///
    /// If you need stronger guarantees, consider using references instead,
    /// as they are statically verified by the borrow-checker to never dangle.
    pub DANGLING_POINTERS_FROM_LOCALS,
    Warn,
    "detects returning a pointer from a local variable"
}

/// FIXME: false negatives (i.e. the lint is not emitted when it should be)
/// 1. Ways to get a temporary that are not recognized:
///    - `owning_temporary.field`
///    - `owning_temporary[index]`
/// 2. No checks for ref-to-ptr conversions:
///    - `&raw [mut] temporary`
///    - `&temporary as *(const|mut) _`
///    - `ptr::from_ref(&temporary)` and friends
#[derive(Clone, Copy, Default)]
pub(crate) struct DanglingPointers;

impl_lint_pass!(DanglingPointers => [DANGLING_POINTERS_FROM_TEMPORARIES, DANGLING_POINTERS_FROM_LOCALS]);

// This skips over const blocks, but they cannot use or return a dangling pointer anyways.
impl<'tcx> LateLintPass<'tcx> for DanglingPointers {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'tcx>,
        fn_decl: &'tcx FnDecl<'tcx>,
        body: &'tcx Body<'tcx>,
        _: Span,
        def_id: LocalDefId,
    ) {
        DanglingPointerSearcher { cx, inside_call_args: false }.visit_body(body);

        if let FnRetTy::Return(ret_ty) = &fn_decl.output
            && let TyKind::Ptr(_) = ret_ty.kind
        {
            // get the return type of the function or closure
            let ty = match cx.tcx.type_of(def_id).instantiate_identity().kind() {
                ty::FnDef(..) => cx.tcx.fn_sig(def_id).instantiate_identity(),
                ty::Closure(_, args) => args.as_closure().sig(),
                _ => return,
            };
            let ty = ty.output();

            // this type is only used for layout computation and pretty-printing, neither of them rely on regions
            let ty = cx.tcx.instantiate_bound_regions_with_erased(ty);

            // verify that we have a pointer type
            let inner_ty = match ty.kind() {
                ty::RawPtr(inner_ty, _) => *inner_ty,
                _ => return,
            };

            if cx
                .tcx
                .layout_of(cx.typing_env().as_query_input(inner_ty))
                .is_ok_and(|layout| !layout.is_1zst())
            {
                let dcx = &DanglingPointerLocalContext {
                    body: def_id,
                    fn_ret: ty,
                    fn_ret_span: ret_ty.span,
                    fn_ret_inner: inner_ty,
                    fn_kind: match fn_kind {
                        FnKind::ItemFn(..) => "function",
                        FnKind::Method(..) => "method",
                        FnKind::Closure => "closure",
                    },
                };

                // look for `return`s
                DanglingPointerReturnSearcher { cx, dcx }.visit_body(body);

                // analyze implicit return expression
                if let ExprKind::Block(block, None) = &body.value.kind
                    && let innermost_block = block.innermost_block()
                    && let Some(expr) = innermost_block.expr
                {
                    lint_addr_of_local(cx, dcx, expr);
                }
            }
        }
    }
}

struct DanglingPointerLocalContext<'tcx> {
    body: LocalDefId,
    fn_ret: Ty<'tcx>,
    fn_ret_span: Span,
    fn_ret_inner: Ty<'tcx>,
    fn_kind: &'static str,
}

struct DanglingPointerReturnSearcher<'lcx, 'tcx> {
    cx: &'lcx LateContext<'tcx>,
    dcx: &'lcx DanglingPointerLocalContext<'tcx>,
}

impl<'tcx> Visitor<'tcx> for DanglingPointerReturnSearcher<'_, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) -> Self::Result {
        if let ExprKind::Ret(Some(expr)) = expr.kind {
            lint_addr_of_local(self.cx, self.dcx, expr);
        }
        walk_expr(self, expr)
    }
}

/// Look for `&<path_to_local_in_same_body>` pattern and emit lint for it
fn lint_addr_of_local<'a>(
    cx: &LateContext<'a>,
    dcx: &DanglingPointerLocalContext<'a>,
    expr: &'a Expr<'a>,
) {
    // peel casts as they do not interest us here, we want the inner expression.
    let (inner, _) = super::utils::peel_casts(cx, expr);

    if let ExprKind::AddrOf(_, _, inner_of) = inner.kind
        && let ExprKind::Path(ref qpath) = inner_of.peel_blocks().kind
        && let Res::Local(from) = cx.qpath_res(qpath, inner_of.hir_id)
        && cx.tcx.hir_enclosing_body_owner(from) == dcx.body
    {
        cx.tcx.emit_node_span_lint(
            DANGLING_POINTERS_FROM_LOCALS,
            expr.hir_id,
            expr.span,
            DanglingPointersFromLocals {
                ret_ty: dcx.fn_ret,
                ret_ty_span: dcx.fn_ret_span,
                fn_kind: dcx.fn_kind,
                local_var: cx.tcx.hir_span(from),
                local_var_name: cx.tcx.hir_ident(from),
                local_var_ty: dcx.fn_ret_inner,
                created_at: (expr.hir_id != inner.hir_id).then_some(inner.span),
            },
        );
    }
}

/// This produces a dangling pointer:
/// ```ignore (example)
/// let ptr = CString::new("hello").unwrap().as_ptr();
/// foo(ptr)
/// ```
///
/// But this does not:
/// ```ignore (example)
/// foo(CString::new("hello").unwrap().as_ptr())
/// ```
///
/// But this does:
/// ```ignore (example)
/// foo({ let ptr = CString::new("hello").unwrap().as_ptr(); ptr })
/// ```
///
/// So we have to keep track of when we are inside of a function/method call argument.
struct DanglingPointerSearcher<'lcx, 'tcx> {
    cx: &'lcx LateContext<'tcx>,
    /// Keeps track of whether we are inside of function/method call arguments,
    /// where this lint should not be emitted.
    ///
    /// See [the main doc][`Self`] for examples.
    inside_call_args: bool,
}

impl Visitor<'_> for DanglingPointerSearcher<'_, '_> {
    fn visit_expr(&mut self, expr: &Expr<'_>) -> Self::Result {
        if !self.inside_call_args {
            lint_expr(self.cx, expr)
        }
        match expr.kind {
            ExprKind::Call(lhs, args) | ExprKind::MethodCall(_, lhs, args, _) => {
                self.visit_expr(lhs);
                self.with_inside_call_args(true, |this| walk_list!(this, visit_expr, args))
            }
            ExprKind::Block(&Block { stmts, expr, .. }, _) => {
                self.with_inside_call_args(false, |this| walk_list!(this, visit_stmt, stmts));
                visit_opt!(self, visit_expr, expr)
            }
            _ => walk_expr(self, expr),
        }
    }
}

impl DanglingPointerSearcher<'_, '_> {
    fn with_inside_call_args<R>(
        &mut self,
        inside_call_args: bool,
        callback: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let old = core::mem::replace(&mut self.inside_call_args, inside_call_args);
        let result = callback(self);
        self.inside_call_args = old;
        result
    }
}

fn lint_expr(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::MethodCall(method, receiver, _args, _span) = expr.kind
        && is_temporary_rvalue(receiver)
        && let ty = cx.typeck_results().expr_ty(receiver)
        && owns_allocation(cx.tcx, ty)
        && let Some(fn_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && find_attr!(cx.tcx.get_all_attrs(fn_id), AttributeKind::AsPtr(_))
    {
        // FIXME: use `emit_node_lint` when `#[primary_span]` is added.
        cx.tcx.emit_node_span_lint(
            DANGLING_POINTERS_FROM_TEMPORARIES,
            expr.hir_id,
            method.ident.span,
            DanglingPointersFromTemporaries {
                callee: method.ident,
                ty,
                ptr_span: method.ident.span,
                temporary_span: receiver.span,
            },
        )
    }
}

fn is_temporary_rvalue(expr: &Expr<'_>) -> bool {
    match expr.kind {
        // Const is not temporary.
        ExprKind::ConstBlock(..) | ExprKind::Repeat(..) | ExprKind::Lit(..) => false,

        // This is literally lvalue.
        ExprKind::Path(..) => false,

        // Calls return rvalues.
        ExprKind::Call(..)
        | ExprKind::MethodCall(..)
        | ExprKind::Use(..)
        | ExprKind::Binary(..) => true,

        // Inner blocks are rvalues.
        ExprKind::If(..) | ExprKind::Loop(..) | ExprKind::Match(..) | ExprKind::Block(..) => true,

        // FIXME: these should probably recurse and typecheck along the way.
        //        Some false negatives are possible for now.
        ExprKind::Index(..) | ExprKind::Field(..) | ExprKind::Unary(..) => false,

        ExprKind::Struct(..) => true,

        // FIXME: this has false negatives, but I do not want to deal with 'static/const promotion just yet.
        ExprKind::Array(..) => false,

        // These typecheck to `!`
        ExprKind::Break(..) | ExprKind::Continue(..) | ExprKind::Ret(..) | ExprKind::Become(..) => {
            false
        }

        // These typecheck to `()`
        ExprKind::Assign(..) | ExprKind::AssignOp(..) | ExprKind::Yield(..) => false,

        // Compiler-magic macros
        ExprKind::AddrOf(..) | ExprKind::OffsetOf(..) | ExprKind::InlineAsm(..) => false,

        // We are not interested in these
        ExprKind::Cast(..)
        | ExprKind::Closure(..)
        | ExprKind::Tup(..)
        | ExprKind::DropTemps(..)
        | ExprKind::Let(..) => false,

        ExprKind::UnsafeBinderCast(..) => false,

        // Not applicable
        ExprKind::Type(..) | ExprKind::Err(..) => false,
    }
}

// Array, Vec, String, CString, MaybeUninit, Cell, Box<[_]>, Box<str>, Box<CStr>, UnsafeCell,
// SyncUnsafeCell, or any of the above in arbitrary many nested Box'es.
fn owns_allocation(tcx: TyCtxt<'_>, ty: Ty<'_>) -> bool {
    if ty.is_array() {
        true
    } else if let Some(inner) = ty.boxed_ty() {
        inner.is_slice()
            || inner.is_str()
            || inner.ty_adt_def().is_some_and(|def| tcx.is_lang_item(def.did(), LangItem::CStr))
            || owns_allocation(tcx, inner)
    } else if let Some(def) = ty.ty_adt_def() {
        for lang_item in [LangItem::String, LangItem::MaybeUninit, LangItem::UnsafeCell] {
            if tcx.is_lang_item(def.did(), lang_item) {
                return true;
            }
        }
        tcx.get_diagnostic_name(def.did()).is_some_and(|name| {
            matches!(name, sym::cstring_type | sym::Vec | sym::Cell | sym::SyncUnsafeCell)
        })
    } else {
        false
    }
}
