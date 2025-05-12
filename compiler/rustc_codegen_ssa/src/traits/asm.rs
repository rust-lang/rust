use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_hir::def_id::DefId;
use rustc_middle::ty::Instance;
use rustc_span::Span;
use rustc_target::asm::InlineAsmRegOrRegClass;

use super::BackendTypes;
use crate::mir::operand::OperandRef;
use crate::mir::place::PlaceRef;

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
    Label {
        label: B::BasicBlock,
    },
}

#[derive(Debug)]
pub enum GlobalAsmOperandRef<'tcx> {
    Const { string: String },
    SymFn { instance: Instance<'tcx> },
    SymStatic { def_id: DefId },
}

pub trait AsmBuilderMethods<'tcx>: BackendTypes {
    /// Take an inline assembly expression and splat it out via LLVM
    fn codegen_inline_asm(
        &mut self,
        template: &[InlineAsmTemplatePiece],
        operands: &[InlineAsmOperandRef<'tcx, Self>],
        options: InlineAsmOptions,
        line_spans: &[Span],
        instance: Instance<'_>,
        dest: Option<Self::BasicBlock>,
        catch_funclet: Option<(Self::BasicBlock, Option<&Self::Funclet>)>,
    );
}

pub trait AsmCodegenMethods<'tcx> {
    fn codegen_global_asm(
        &mut self,
        template: &[InlineAsmTemplatePiece],
        operands: &[GlobalAsmOperandRef<'tcx>],
        options: InlineAsmOptions,
        line_spans: &[Span],
    );

    /// The mangled name of this instance
    ///
    /// Additional mangling is used on
    /// some targets to add a leading underscore (Mach-O)
    /// or byte count suffixes (x86 Windows).
    fn mangled_name(&self, instance: Instance<'tcx>) -> String;
}
