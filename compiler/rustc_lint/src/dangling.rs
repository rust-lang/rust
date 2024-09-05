use rustc_hir::{Block, Expr, ExprKind, HirId, LangItem};
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::symbol::sym;

use crate::lints::InstantlyDangling;
use crate::{LateContext, LateLintPass, LintContext};

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

#[derive(Clone, Debug, PartialEq, Eq)]
enum LifetimeExtension {
    /// Lifetime extension has not kicked in yet, but it will soon.
    /// Example: walking LHS of a function/method call.
    EnableLater { after_exit: HirId, until_exit: HirId },
    /// Lifetime extension is currently active.
    /// Example: walking a function/method call's arguments.
    Enable { until_exit: HirId },
    /// Temporary disable lifetime extension.
    /// Example: statements of a block that is a function/method call's argument.
    Disable { until_exit: HirId },
}

#[derive(Clone, Default)]
pub(crate) struct DanglingPointers {
    /// Trying to deal with argument lifetime extension.
    ///
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
    /// We have to deal with this situation somehow.
    ///
    /// If we were a visitor, we could just keep track of
    /// when we enter and exit places where lifetime extension kicks in
    /// during visiting/walking and update a boolean flag accordingly.
    ///
    /// But we are not a visitor. We are a LateLintPass.
    /// We are not the one who does the visiting & walking
    /// and can maintain this state directly in the call stack.
    /// But we do get called on every expression there is,
    /// both when entering it and exiting from it
    /// during our depth-first walk of the tree.
    /// So let's try to maintain this context stack explicitly
    /// instead of as a part of the call stack.
    nested_calls: Vec<LifetimeExtension>,
}

impl_lint_pass!(DanglingPointers => [DANGLING_POINTERS_FROM_TEMPORARIES]);

/// FIXME: false negatives (i.e. the lint is not emitted when it should be)
/// 1. Method calls that are not checked for:
///    - [`temporary_unsafe_cell.get()`][`core::cell::UnsafeCell::get()`]
///    - [`temporary_sync_unsafe_cell.get()`][`core::cell::SyncUnsafeCell::get()`]
/// 2. Ways to get a temporary that are not recognized:
///    - `owning_temporary.field`
///    - `owning_temporary[index]`
/// 3. No checks for ref-to-ptr conversions:
///    - `&raw [mut] temporary`
///    - `&temporary as *(const|mut) _`
///    - `ptr::from_ref(&temporary)` and friends
impl<'tcx> LateLintPass<'tcx> for DanglingPointers {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let Some(LifetimeExtension::Enable { .. }) = self.nested_calls.last() {
            match expr.kind {
                ExprKind::Block(Block { stmts: [.., last_stmt], .. }, _) => self
                    .nested_calls
                    .push(LifetimeExtension::Disable { until_exit: last_stmt.hir_id }),
                _ => {
                    tracing::debug!(skip = ?cx.sess().source_map().span_to_snippet(expr.span));
                    return;
                }
            }
        }

        lint_expr(cx, expr);

        if let ExprKind::Call(lhs, _args) | ExprKind::MethodCall(_, lhs, _args, _) = expr.kind {
            self.nested_calls.push(LifetimeExtension::EnableLater {
                after_exit: lhs.hir_id,
                until_exit: expr.hir_id,
            })
        }
    }

    fn check_expr_post(&mut self, _: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        self.nested_calls.pop_if(|pos| match pos {
            LifetimeExtension::Enable { until_exit }
            | LifetimeExtension::Disable { until_exit } => expr.hir_id == *until_exit,

            &mut LifetimeExtension::EnableLater { after_exit, until_exit } => {
                if expr.hir_id == after_exit {
                    *pos = LifetimeExtension::Enable { until_exit };
                };
                false
            }
        });
    }
}

fn lint_expr(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::MethodCall(method, receiver, _args, _span) = expr.kind
        && matches!(method.ident.name, sym::as_ptr | sym::as_mut_ptr)
        && is_temporary_rvalue(receiver)
        && let ty = cx.typeck_results().expr_ty(receiver)
        && is_interesting(cx.tcx, ty)
    {
        cx.emit_span_lint(
            DANGLING_POINTERS_FROM_TEMPORARIES,
            method.ident.span,
            InstantlyDangling {
                callee: method.ident.name,
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
        ExprKind::Call(..) | ExprKind::MethodCall(..) | ExprKind::Binary(..) => true,

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

        // Not applicable
        ExprKind::Type(..) | ExprKind::Err(..) => false,
    }
}

// Array, Vec, String, CString, MaybeUninit, Cell, Box<[_]>, Box<str>, Box<CStr>,
// or any of the above in arbitrary many nested Box'es.
fn is_interesting(tcx: TyCtxt<'_>, ty: Ty<'_>) -> bool {
    if ty.is_array() {
        true
    } else if let Some(inner) = ty.boxed_ty() {
        inner.is_slice()
            || inner.is_str()
            || inner.ty_adt_def().is_some_and(|def| tcx.is_lang_item(def.did(), LangItem::CStr))
            || is_interesting(tcx, inner)
    } else if let Some(def) = ty.ty_adt_def() {
        for lang_item in [LangItem::String, LangItem::MaybeUninit] {
            if tcx.is_lang_item(def.did(), lang_item) {
                return true;
            }
        }
        tcx.get_diagnostic_name(def.did())
            .is_some_and(|name| matches!(name, sym::cstring_type | sym::Vec | sym::Cell))
    } else {
        false
    }
}
