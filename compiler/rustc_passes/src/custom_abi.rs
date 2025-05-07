use rustc_abi::ExternAbi;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{LocalDefId, LocalModDefId};
use rustc_hir::intravisit::Visitor;
use rustc_hir::{self as hir, ExprKind, FnSig};
use rustc_middle::hir::nested_filter::OnlyBodies;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::{BytePos, sym};

use crate::errors::{
    AbiCustomCall, AbiCustomClothedFunction, AbiCustomSafeForeignFunction, AbiCustomSafeFunction,
};

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

        // An `extern "custom"` function cannot be marked as `safe`.
        if let DefKind::ForeignMod = def_kind
            && let hir::Node::Item(item) = tcx.hir_node_by_def_id(def_id)
            && let hir::ItemKind::ForeignMod { abi: ExternAbi::Custom, items } = item.kind
        {
            for item in items {
                let hir_id = item.id.hir_id();
                if let hir::Node::ForeignItem(foreign_item) = tcx.hir_node(hir_id)
                    && let hir::ForeignItemKind::Fn(sig, _, _) = foreign_item.kind
                    && sig.header.is_safe()
                {
                    let len = "safe ".len() as u32;
                    let safe_span = sig.span.shrink_to_lo().with_hi(sig.span.lo() + BytePos(len));
                    tcx.dcx().emit_err(AbiCustomSafeForeignFunction { span: sig.span, safe_span });
                }
            }
        }

        // An `extern "custom"` function cannot be a `const fn`, because `naked_asm!` cannot be
        // evaluated at compile time, and `extern` blocks cannot declare `const fn` functions.
        // Therefore, to find all calls to `extern "custom"` functions, it suffices to traverse
        // all function bodies (i.e. we can skip `const` and `static` initializers).
        if !matches!(def_kind, DefKind::Fn | DefKind::AssocFn) {
            continue;
        }

        let body = match tcx.hir_node_by_def_id(def_id) {
            hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Fn { sig, body: body_id, .. },
                ..
            })
            | hir::Node::ImplItem(hir::ImplItem {
                kind: hir::ImplItemKind::Fn(sig, body_id),
                ..
            }) => {
                check_signature(tcx, def_id, sig, true);
                tcx.hir_body(*body_id)
            }
            hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(sig, trait_fn),
                ..
            }) => match trait_fn {
                hir::TraitFn::Required(_) => {
                    check_signature(tcx, def_id, sig, false);
                    continue;
                }
                hir::TraitFn::Provided(body_id) => {
                    check_signature(tcx, def_id, sig, true);
                    tcx.hir_body(*body_id)
                }
            },
            _ => continue,
        };

        let mut visitor = CheckCustomAbi { tcx };
        visitor.visit_body(body);
    }
}

fn check_signature<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, sig: &FnSig<'tcx>, has_body: bool) {
    if sig.header.abi == ExternAbi::Custom {
        // Function definitions that use `extern "custom"` must be naked functions.
        if has_body && !tcx.has_attr(def_id, sym::naked) {
            tcx.dcx().emit_err(AbiCustomClothedFunction {
                span: sig.span,
                naked_span: sig.span.shrink_to_lo(),
            });
        }

        // Function definitions that use `extern "custom"` must unsafe.
        if sig.header.is_safe() {
            tcx.dcx().emit_err(AbiCustomSafeFunction {
                span: sig.span,
                unsafe_span: sig.span.shrink_to_lo(),
            });
        }
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
