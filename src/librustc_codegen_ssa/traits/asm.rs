use super::BackendTypes;
use crate::mir::operand::OperandRef;
use crate::mir::place::PlaceRef;
use rustc_ast::ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_hir::def_id::DefId;
use rustc_hir::{GlobalAsm, LlvmInlineAsmInner};
use rustc_middle::ty::Instance;
use rustc_span::Span;
use rustc_target::asm::InlineAsmRegOrRegClass;

#[derive(Debug)]
pub enum InlineAsmOperandRef<'tcx, B: BackendTypes + ?Sized> {
    In {
        reg: InlineAsmRegOrRegClass,
        value: OperandRef<'tcx, B::Value>,
    },
    Out {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        place: Option<PlaceRef<'tcx, B::Value>>,
    },
    InOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        in_value: OperandRef<'tcx, B::Value>,
        out_place: Option<PlaceRef<'tcx, B::Value>>,
    },
    Const {
        string: String,
    },
    SymFn {
        instance: Instance<'tcx>,
    },
    SymStatic {
        def_id: DefId,
    },
}

pub trait AsmBuilderMethods<'tcx>: BackendTypes {
    /// Take an inline assembly expression and splat it out via LLVM
    fn codegen_llvm_inline_asm(
        &mut self,
        ia: &LlvmInlineAsmInner,
        outputs: Vec<PlaceRef<'tcx, Self::Value>>,
        inputs: Vec<Self::Value>,
        span: Span,
    ) -> bool;

    /// Take an inline assembly expression and splat it out via LLVM
    fn codegen_inline_asm(
        &mut self,
        template: &[InlineAsmTemplatePiece],
        operands: &[InlineAsmOperandRef<'tcx, Self>],
        options: InlineAsmOptions,
        span: Span,
    );
}

pub trait AsmMethods {
    fn codegen_global_asm(&self, ga: &GlobalAsm);
}
