use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::InlineAsmOptions;
use rustc_hir::{Expr, ExprKind, InlineAsm, InlineAsmOperand};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks if any pointer is being passed to an asm! block with `nomem` option.
    ///
    /// ### Why is this bad?
    /// `nomem` forbids any reads or writes to memory and passing a pointer suggests
    /// that either of those will happen.
    ///
    /// ### Example
    /// ```no_run
    /// fn f(p: *mut u32) {
    ///     unsafe { core::arch::asm!("mov [{p}], 42", p = in(reg) p, options(nomem, nostack)); }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn f(p: *mut u32) {
    ///     unsafe { core::arch::asm!("mov [{p}], 42", p = in(reg) p, options(nostack)); }
    /// }
    /// ```
    #[clippy::version = "1.81.0"]
    pub POINTERS_IN_NOMEM_ASM_BLOCK,
    suspicious,
    "pointers in nomem asm block"
}

declare_lint_pass!(PointersInNomemAsmBlock => [POINTERS_IN_NOMEM_ASM_BLOCK]);

impl<'tcx> LateLintPass<'tcx> for PointersInNomemAsmBlock {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if let ExprKind::InlineAsm(asm) = &expr.kind {
            check_asm(cx, asm);
        }
    }
}

fn check_asm(cx: &LateContext<'_>, asm: &InlineAsm<'_>) {
    if !asm.options.contains(InlineAsmOptions::NOMEM) {
        return;
    }

    let spans = asm
        .operands
        .iter()
        .filter(|(op, _span)| has_in_operand_pointer(cx, op))
        .map(|(_op, span)| *span)
        .collect::<Vec<Span>>();

    if spans.is_empty() {
        return;
    }

    span_lint_and_then(
        cx,
        POINTERS_IN_NOMEM_ASM_BLOCK,
        spans,
        "passing pointers to nomem asm block",
        additional_notes,
    );
}

fn has_in_operand_pointer(cx: &LateContext<'_>, asm_op: &InlineAsmOperand<'_>) -> bool {
    let asm_in_expr = match asm_op {
        InlineAsmOperand::SymStatic { .. }
        | InlineAsmOperand::Out { .. }
        | InlineAsmOperand::Const { .. }
        | InlineAsmOperand::SymFn { .. }
        | InlineAsmOperand::Label { .. } => return false,
        InlineAsmOperand::SplitInOut { in_expr, .. } => in_expr,
        InlineAsmOperand::In { expr, .. } | InlineAsmOperand::InOut { expr, .. } => expr,
    };

    // This checks for raw ptrs, refs and function pointers - the last one
    // also technically counts as reading memory.
    cx.typeck_results().expr_ty(asm_in_expr).is_any_ptr()
}

fn additional_notes(diag: &mut rustc_errors::Diag<'_, ()>) {
    diag.note("`nomem` means that no memory write or read happens inside the asm! block");
    diag.note("if this is intentional and no pointers are read or written to, consider allowing the lint");
}
