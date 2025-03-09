use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Body, Location, Operand, Terminator, TerminatorKind};
use rustc_middle::ty::{TyCtxt, UintTy};
use rustc_session::lint::builtin::REDUNDANT_TRANSMUTATION;
use rustc_span::source_map::Spanned;
use rustc_span::{Span, sym};
use rustc_type_ir::TyKind::*;

use crate::errors::RedundantTransmute as Error;

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
    fn is_redundant_transmute(
        &self,
        function: &Operand<'tcx>,
        arg: String,
        span: Span,
    ) -> Option<Error> {
        let fn_sig = function.ty(self.body, self.tcx).fn_sig(self.tcx).skip_binder();
        let [input] = fn_sig.inputs() else { return None };

        let err = |sugg| Error { span, sugg, help: None };

        Some(match (input.kind(), fn_sig.output().kind()) {
            // dont check the length; transmute does that for us.
            // [u8; _] => primitive
            (Array(t, _), Uint(_) | Float(_) | Int(_)) if *t.kind() == Uint(UintTy::U8) => Error {
                sugg: format!("{}::from_ne_bytes({arg})", fn_sig.output()),
                help: Some(
                    "there's also `from_le_bytes` and `from_ne_bytes` if you expect a particular byte order",
                ),
                span,
            },
            // primitive => [u8; _]
            (Uint(_) | Float(_) | Int(_), Array(t, _)) if *t.kind() == Uint(UintTy::U8) => Error {
                sugg: format!("{input}::to_ne_bytes({arg})"),
                help: Some(
                    "there's also `to_le_bytes` and `to_ne_bytes` if you expect a particular byte order",
                ),
                span,
            },
            // char → u32
            (Char, Uint(UintTy::U32)) => err(format!("u32::from({arg})")),
            // u32 → char
            (Uint(UintTy::U32), Char) => Error {
                sugg: format!("char::from_u32_unchecked({arg})"),
                help: Some("consider `char::from_u32(…).unwrap()`"),
                span,
            },
            // uNN → iNN
            (Uint(ty), Int(_)) => err(format!("{}::cast_signed({arg})", ty.name_str())),
            // iNN → uNN
            (Int(ty), Uint(_)) => err(format!("{}::cast_unsigned({arg})", ty.name_str())),
            // fNN → uNN
            (Float(ty), Uint(..)) => err(format!("{}::to_bits({arg})", ty.name_str())),
            // uNN → fNN
            (Uint(_), Float(ty)) => err(format!("{}::from_bits({arg})", ty.name_str())),
            // bool → { x8 }
            (Bool, Int(..) | Uint(..)) => err(format!("({arg}) as {}", fn_sig.output())),
            // u8 → bool
            (Uint(_), Bool) => err(format!("({arg} == 1)")),
            _ => return None,
        })
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
