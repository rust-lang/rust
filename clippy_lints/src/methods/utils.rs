use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{get_parent_expr, path_to_local_id, usage};
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::{BorrowKind, Expr, ExprKind, HirId, Mutability, Pat};
use rustc_lint::LateContext;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::sym;

pub(super) fn derefs_to_slice<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    ty: Ty<'tcx>,
) -> Option<&'tcx hir::Expr<'tcx>> {
    fn may_slice<'a>(cx: &LateContext<'a>, ty: Ty<'a>) -> bool {
        match ty.kind() {
            ty::Slice(_) => true,
            ty::Adt(def, _) if def.is_box() => may_slice(cx, ty.boxed_ty()),
            ty::Adt(..) => is_type_diagnostic_item(cx, ty, sym::Vec),
            ty::Array(_, size) => size.try_eval_target_usize(cx.tcx, cx.param_env).is_some(),
            ty::Ref(_, inner, _) => may_slice(cx, *inner),
            _ => false,
        }
    }

    if let hir::ExprKind::MethodCall(path, self_arg, ..) = &expr.kind {
        if path.ident.name == sym::iter && may_slice(cx, cx.typeck_results().expr_ty(self_arg)) {
            Some(self_arg)
        } else {
            None
        }
    } else {
        match ty.kind() {
            ty::Slice(_) => Some(expr),
            ty::Adt(def, _) if def.is_box() && may_slice(cx, ty.boxed_ty()) => Some(expr),
            ty::Ref(_, inner, _) => {
                if may_slice(cx, *inner) {
                    Some(expr)
                } else {
                    None
                }
            },
            _ => None,
        }
    }
}

pub(super) fn get_hint_if_single_char_arg(
    cx: &LateContext<'_>,
    arg: &hir::Expr<'_>,
    applicability: &mut Applicability,
) -> Option<String> {
    if_chain! {
        if let hir::ExprKind::Lit(lit) = &arg.kind;
        if let ast::LitKind::Str(r, style) = lit.node;
        let string = r.as_str();
        if string.chars().count() == 1;
        then {
            let snip = snippet_with_applicability(cx, arg.span, string, applicability);
            let ch = if let ast::StrStyle::Raw(nhash) = style {
                let nhash = nhash as usize;
                // for raw string: r##"a"##
                &snip[(nhash + 2)..(snip.len() - 1 - nhash)]
            } else {
                // for regular string: "a"
                &snip[1..(snip.len() - 1)]
            };

            let hint = format!("'{}'", match ch {
                "'" => "\\'" ,
                r"\" => "\\\\",
                _ => ch,
            });

            Some(hint)
        } else {
            None
        }
    }
}

/// The core logic of `check_for_loop_iter` in `unnecessary_iter_cloned.rs`, this function wraps a
/// use of `CloneOrCopyVisitor`.
pub(super) fn clone_or_copy_needed<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &Pat<'tcx>,
    body: &'tcx Expr<'tcx>,
) -> (bool, Vec<&'tcx Expr<'tcx>>) {
    let mut visitor = CloneOrCopyVisitor {
        cx,
        binding_hir_ids: pat_bindings(pat),
        clone_or_copy_needed: false,
        addr_of_exprs: Vec::new(),
    };
    visitor.visit_expr(body);
    (visitor.clone_or_copy_needed, visitor.addr_of_exprs)
}

/// Returns a vector of all `HirId`s bound by the pattern.
fn pat_bindings(pat: &Pat<'_>) -> Vec<HirId> {
    let mut collector = usage::ParamBindingIdCollector {
        binding_hir_ids: Vec::new(),
    };
    collector.visit_pat(pat);
    collector.binding_hir_ids
}

/// `clone_or_copy_needed` will be false when `CloneOrCopyVisitor` is done visiting if the only
/// operations performed on `binding_hir_ids` are:
/// * to take non-mutable references to them
/// * to use them as non-mutable `&self` in method calls
/// If any of `binding_hir_ids` is used in any other way, then `clone_or_copy_needed` will be true
/// when `CloneOrCopyVisitor` is done visiting.
struct CloneOrCopyVisitor<'cx, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    binding_hir_ids: Vec<HirId>,
    clone_or_copy_needed: bool,
    addr_of_exprs: Vec<&'tcx Expr<'tcx>>,
}

impl<'cx, 'tcx> Visitor<'tcx> for CloneOrCopyVisitor<'cx, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        walk_expr(self, expr);
        if self.is_binding(expr) {
            if let Some(parent) = get_parent_expr(self.cx, expr) {
                match parent.kind {
                    ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, _) => {
                        self.addr_of_exprs.push(parent);
                        return;
                    },
                    ExprKind::MethodCall(.., args, _) => {
                        if_chain! {
                            if args.iter().all(|arg| !self.is_binding(arg));
                            if let Some(method_def_id) = self.cx.typeck_results().type_dependent_def_id(parent.hir_id);
                            let method_ty = self.cx.tcx.bound_type_of(method_def_id).subst_identity();
                            let self_ty = method_ty.fn_sig(self.cx.tcx).input(0).skip_binder();
                            if matches!(self_ty.kind(), ty::Ref(_, _, Mutability::Not));
                            then {
                                return;
                            }
                        }
                    },
                    _ => {},
                }
            }
            self.clone_or_copy_needed = true;
        }
    }
}

impl<'cx, 'tcx> CloneOrCopyVisitor<'cx, 'tcx> {
    fn is_binding(&self, expr: &Expr<'tcx>) -> bool {
        self.binding_hir_ids
            .iter()
            .any(|hir_id| path_to_local_id(expr, *hir_id))
    }
}
