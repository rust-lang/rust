use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Body, Location, Operand, Terminator, TerminatorKind};
use rustc_middle::ty::*;
use rustc_session::lint::builtin::UNNECESSARY_TRANSMUTES;
use rustc_span::source_map::Spanned;
use rustc_span::{Span, sym};

use crate::errors::UnnecessaryTransmute as Error;

/// Check for transmutes that overlap with stdlib methods.
/// For example, transmuting `[u8; 4]` to `u32`.
/// We chose not to lint u8 -> bool transmutes, see #140431
pub(super) struct CheckUnnecessaryTransmutes;

impl<'tcx> crate::MirLint<'tcx> for CheckUnnecessaryTransmutes {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let mut checker = UnnecessaryTransmuteChecker { body, tcx };
        checker.visit_body(body);
    }
}

struct UnnecessaryTransmuteChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
}

impl<'a, 'tcx> UnnecessaryTransmuteChecker<'a, 'tcx> {
    fn is_unnecessary_transmute(
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
                    "there's also `from_le_bytes` and `from_be_bytes` if you expect a particular byte order",
                ),
                span,
            },
            // primitive => [u8; _]
            (Uint(_) | Float(_) | Int(_), Array(t, _)) if *t.kind() == Uint(UintTy::U8) => Error {
                sugg: format!("{input}::to_ne_bytes({arg})"),
                help: Some(
                    "there's also `to_le_bytes` and `to_be_bytes` if you expect a particular byte order",
                ),
                span,
            },
            // char → u32
            (Char, Uint(UintTy::U32)) => err(format!("u32::from({arg})")),
            // char (→ u32) → i32
            (Char, Int(IntTy::I32)) => err(format!("u32::from({arg}).cast_signed()")),
            // u32 → char
            (Uint(UintTy::U32), Char) => Error {
                sugg: format!("char::from_u32_unchecked({arg})"),
                help: Some("consider `char::from_u32(…).unwrap()`"),
                span,
            },
            // i32 → char
            (Int(IntTy::I32), Char) => Error {
                sugg: format!("char::from_u32_unchecked(i32::cast_unsigned({arg}))"),
                help: Some("consider `char::from_u32(i32::cast_unsigned(…)).unwrap()`"),
                span,
            },
            // uNN → iNN
            (Uint(ty), Int(_)) => err(format!("{}::cast_signed({arg})", ty.name_str())),
            // iNN → uNN
            (Int(ty), Uint(_)) => err(format!("{}::cast_unsigned({arg})", ty.name_str())),
            // fNN → xsize
            (Float(ty), Uint(UintTy::Usize)) => {
                err(format!("{}::to_bits({arg}) as usize", ty.name_str()))
            }
            (Float(ty), Int(IntTy::Isize)) => {
                err(format!("{}::to_bits({arg}) as isize", ty.name_str()))
            }
            // fNN (→ uNN) → iNN
            (Float(ty), Int(..)) => err(format!("{}::to_bits({arg}).cast_signed()", ty.name_str())),
            // fNN → uNN
            (Float(ty), Uint(..)) => err(format!("{}::to_bits({arg})", ty.name_str())),
            // xsize → fNN
            (Uint(UintTy::Usize) | Int(IntTy::Isize), Float(ty)) => {
                err(format!("{}::from_bits({arg} as _)", ty.name_str(),))
            }
            // iNN (→ uNN) → fNN
            (Int(int_ty), Float(ty)) => err(format!(
                "{}::from_bits({}::cast_unsigned({arg}))",
                ty.name_str(),
                int_ty.name_str()
            )),
            // uNN → fNN
            (Uint(_), Float(ty)) => err(format!("{}::from_bits({arg})", ty.name_str())),
            // bool → { x8 }
            (Bool, Int(..) | Uint(..)) => err(format!("({arg}) as {}", fn_sig.output())),
            _ => return None,
        })
    }
}

impl<'tcx> Visitor<'tcx> for UnnecessaryTransmuteChecker<'_, 'tcx> {
    // Check each block's terminator for calls to pointer to integer transmutes
    // in const functions or associated constants and emit a lint.
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        if let TerminatorKind::Call { func, args, .. } = &terminator.kind
            && let [Spanned { span: arg, .. }] = **args
            && let Some((func_def_id, _)) = func.const_fn_def()
            && self.tcx.is_intrinsic(func_def_id, sym::transmute)
            && let span = self.body.source_info(location).span
            && let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(arg)
            && let Some(lint) = self.is_unnecessary_transmute(func, snippet, span)
            && let Some(hir_id) = terminator.source_info.scope.lint_root(&self.body.source_scopes)
        {
            self.tcx.emit_node_span_lint(UNNECESSARY_TRANSMUTES, hir_id, span, lint);
        }
    }
}
