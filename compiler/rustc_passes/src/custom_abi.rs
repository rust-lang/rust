use rustc_abi::ExternAbi;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalModDefId;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{self as hir, ExprKind};
use rustc_middle::hir::nested_filter::OnlyBodies;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::sym;

use crate::errors::{AbiCustomCall, AbiCustomClothedFunction};

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_custom_abi, ..*providers };
}

fn check_mod_custom_abi(tcx: TyCtxt<'_>, module_def_id: LocalModDefId) {
    if !tcx.features().abi_custom() {
        return;
    }

    let items = tcx.hir_module_items(module_def_id);
    for def_id in items.definitions() {
        let def_kind = tcx.def_kind(def_id);

        // An `extern "custom"` function cannot be a `const fn`, because `naked_asm!` cannot be
        // evaluated at compile time, and `extern` blocks cannot declare `const fn` functions.
        // Therefore, to find all calls to `extern "custom"` functions, it suffices to traverse
        // all function bodies (i.e. we can skip `const` and `static` initializers).
        if !matches!(def_kind, DefKind::Fn | DefKind::AssocFn) {
            continue;
        }

        let (sig, body) = match tcx.hir_node_by_def_id(def_id) {
            hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Fn { sig, body: body_id, .. },
                ..
            })
            | hir::Node::ImplItem(hir::ImplItem {
                kind: hir::ImplItemKind::Fn(sig, body_id),
                ..
            })
            | hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(sig, hir::TraitFn::Provided(body_id)),
                ..
            }) => (sig, tcx.hir_body(*body_id)),
            _ => continue,
        };

        if sig.header.abi == ExternAbi::Custom {
            // Function definitions that use `extern "custom"` must be naked functions.
            if !tcx.has_attr(def_id, sym::naked) {
                tcx.dcx().emit_err(AbiCustomClothedFunction {
                    span: sig.span,
                    naked_span: sig.span.shrink_to_lo(),
                });
            }
        }

        let mut visitor = CheckCustomAbi { tcx };
        visitor.visit_body(body);
    }
}

struct CheckCustomAbi<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> Visitor<'tcx> for CheckCustomAbi<'tcx> {
    type NestedFilter = OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        let opt_span_and_abi = match expr.kind {
            ExprKind::Call(fun, _) => {
                let fun_ty = self.tcx.typeck(fun.hir_id.owner.def_id).node_type(fun.hir_id);

                match *fun_ty.kind() {
                    ty::FnDef(def_id, _) => {
                        let sig = self.tcx.fn_sig(def_id).skip_binder().skip_binder();
                        Some((expr.span, sig.abi))
                    }
                    ty::FnPtr(_, header) => Some((expr.span, header.abi)),
                    _ => None,
                }
            }

            ExprKind::MethodCall(_, receiver, _, span) => {
                let opt_def_id = self
                    .tcx
                    .typeck(receiver.hir_id.owner.def_id)
                    .type_dependent_def_id(expr.hir_id);

                opt_def_id.map(|def_id| {
                    let sig = self.tcx.fn_sig(def_id).skip_binder().skip_binder();
                    (span, sig.abi)
                })
            }
            _ => None,
        };

        if let Some((span, ExternAbi::Custom)) = opt_span_and_abi {
            self.tcx.dcx().emit_err(AbiCustomCall { span });
        }

        hir::intravisit::walk_expr(self, expr);
    }
}
