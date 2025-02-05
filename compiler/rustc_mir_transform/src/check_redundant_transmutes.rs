use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Body, Location, Operand, Terminator, TerminatorKind};
use rustc_middle::ty::{Ty, TyCtxt, UintTy};
use rustc_session::lint::builtin::REDUNDANT_TRANSMUTATION;
use rustc_span::source_map::Spanned;
use rustc_span::{Span, sym};
use rustc_type_ir::TyKind::*;

use crate::errors;

/// Check for transmutes that overlap with stdlib methods.
/// For example, transmuting `[u8; 4]` to `u32`.
pub(super) struct CheckRedundantTransmutes;

impl<'tcx> crate::MirLint<'tcx> for CheckRedundantTransmutes {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let mut checker = RedundantTransmutesChecker { body, tcx };
        checker.visit_body(body);
    }
}

struct RedundantTransmutesChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
}

impl<'a, 'tcx> RedundantTransmutesChecker<'a, 'tcx> {
    /// This functions checks many things:
    /// 1. if the source (`transmute::<$x, _>`) is `[u8; _]`, check if the output is a `{uif}xx`
    /// 2. swap and do the same check.
    /// 3. in the case of `char` → `u32` suggest `to_u32` and `from_u32_unchecked`
    /// 4. `uxx` → `ixx` should be `as`
    /// Returns the replacement.
    fn is_redundant_transmute(
        &self,
        function: &Operand<'tcx>,
        arg: String,
        span: Span,
    ) -> Option<errors::RedundantTransmute> {
        let fn_sig = function.ty(self.body, self.tcx).fn_sig(self.tcx).skip_binder();
        let [input] = fn_sig.inputs() else { return None };

        // Checks if `x` is `[u8; _]`
        let is_u8s = |x: Ty<'tcx>| matches!(x.kind(), Array(t, _) if *t.kind() == Uint(UintTy::U8));
        // dont check the length; transmute does that for us.
        if is_u8s(*input) && matches!(fn_sig.output().kind(), Uint(..) | Float(_) | Int(_)) {
            // FIXME: get the whole expression out?
            return Some(errors::RedundantTransmute {
                sugg: format!("{}::from_ne_bytes({arg})", fn_sig.output()),
                help: Some(
                    "there's also `from_le_bytes` and `from_ne_bytes` if you expect a particular byte order",
                ),
                span,
            });
        }
        if is_u8s(fn_sig.output()) && matches!(input.kind(), Uint(..) | Float(_) | Int(_)) {
            return Some(errors::RedundantTransmute {
                sugg: format!("{input}::to_ne_bytes({arg})"),
                help: Some(
                    "there's also `to_le_bytes` and `to_ne_bytes` if you expect a particular byte order",
                ),
                span,
            });
        }
        return match input.kind() {
            // char → u32
            Char => matches!(fn_sig.output().kind(), Uint(UintTy::U32)).then(|| {
                errors::RedundantTransmute { sugg: format!("({arg}) as u32"), help: None, span }
            }),
            // u32 → char
            Uint(UintTy::U32) if *fn_sig.output().kind() == Char => {
                Some(errors::RedundantTransmute {
                    sugg: format!("char::from_u32_unchecked({arg})"),
                    help: Some("consider `char::from_u32(…).unwrap()`"),
                    span,
                })
            }
            // bool → (uNN ↔ iNN)
            Bool | Uint(..) | Int(..) => {
                matches!(fn_sig.output().kind(), Int(..) | Uint(..)).then(|| {
                    errors::RedundantTransmute {
                        sugg: format!("({arg}) as {}", fn_sig.output()),
                        help: None,
                        span,
                    }
                })
            }
            _ => None,
        };
    }
}

impl<'tcx> Visitor<'tcx> for RedundantTransmutesChecker<'_, 'tcx> {
    // Check each block's terminator for calls to pointer to integer transmutes
    // in const functions or associated constants and emit a lint.
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        if let TerminatorKind::Call { func, args, .. } = &terminator.kind
            && let [Spanned { span: arg, .. }] = **args
            && let Some((func_def_id, _)) = func.const_fn_def()
            && self.tcx.is_intrinsic(func_def_id, sym::transmute)
            && let span = self.body.source_info(location).span
            && let Some(lint) = self.is_redundant_transmute(
                func,
                self.tcx.sess.source_map().span_to_snippet(arg).expect("ok"),
                span,
            )
            && let Some(call_id) = self.body.source.def_id().as_local()
        {
            let hir_id = self.tcx.local_def_id_to_hir_id(call_id);

            self.tcx.emit_node_span_lint(REDUNDANT_TRANSMUTATION, hir_id, span, lint);
        }
    }
}
