use std::fmt;

use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::ast::{Expr, ExprKind, InlineAsmOptions};
use rustc_ast::{InlineAsm, Item, ItemKind};
use rustc_lint::{EarlyContext, EarlyLintPass, Lint, LintContext};
use rustc_session::declare_lint_pass;
use rustc_span::Span;
use rustc_target::asm::InlineAsmArch;

#[derive(Clone, Copy, PartialEq, Eq)]
enum AsmStyle {
    Intel,
    Att,
}

impl fmt::Display for AsmStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AsmStyle::Intel => f.write_str("Intel"),
            AsmStyle::Att => f.write_str("AT&T"),
        }
    }
}

impl std::ops::Not for AsmStyle {
    type Output = AsmStyle;

    fn not(self) -> AsmStyle {
        match self {
            AsmStyle::Intel => AsmStyle::Att,
            AsmStyle::Att => AsmStyle::Intel,
        }
    }
}

fn check_asm_syntax(
    lint: &'static Lint,
    cx: &EarlyContext<'_>,
    inline_asm: &InlineAsm,
    span: Span,
    check_for: AsmStyle,
) {
    if matches!(cx.sess().asm_arch, Some(InlineAsmArch::X86 | InlineAsmArch::X86_64)) {
        let style = if inline_asm.options.contains(InlineAsmOptions::ATT_SYNTAX) {
            AsmStyle::Att
        } else {
            AsmStyle::Intel
        };

        if style == check_for {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(cx, lint, span, format!("{style} x86 assembly syntax used"), |diag| {
                diag.help(format!("use {} x86 assembly syntax", !style));
            });
        }
    }
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of Intel x86 assembly syntax.
    ///
    /// ### Why restrict this?
    /// To enforce consistent use of AT&T x86 assembly syntax.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// # #![feature(asm)]
    /// # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// # unsafe { let ptr = "".as_ptr();
    /// # use std::arch::asm;
    /// asm!("lea {}, [{}]", lateout(reg) _, in(reg) ptr);
    /// # }
    /// ```
    /// Use instead:
    /// ```rust,no_run
    /// # #![feature(asm)]
    /// # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// # unsafe { let ptr = "".as_ptr();
    /// # use std::arch::asm;
    /// asm!("lea ({}), {}", in(reg) ptr, lateout(reg) _, options(att_syntax));
    /// # }
    /// ```
    #[clippy::version = "1.49.0"]
    pub INLINE_ASM_X86_INTEL_SYNTAX,
    restriction,
    "prefer AT&T x86 assembly syntax"
}

declare_lint_pass!(InlineAsmX86IntelSyntax => [INLINE_ASM_X86_INTEL_SYNTAX]);

impl EarlyLintPass for InlineAsmX86IntelSyntax {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::InlineAsm(inline_asm) = &expr.kind {
            check_asm_syntax(INLINE_ASM_X86_INTEL_SYNTAX, cx, inline_asm, expr.span, AsmStyle::Intel);
        }
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if let ItemKind::GlobalAsm(inline_asm) = &item.kind {
            check_asm_syntax(INLINE_ASM_X86_INTEL_SYNTAX, cx, inline_asm, item.span, AsmStyle::Intel);
        }
    }
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of AT&T x86 assembly syntax.
    ///
    /// ### Why restrict this?
    /// To enforce consistent use of Intel x86 assembly syntax.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// # #![feature(asm)]
    /// # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// # unsafe { let ptr = "".as_ptr();
    /// # use std::arch::asm;
    /// asm!("lea ({}), {}", in(reg) ptr, lateout(reg) _, options(att_syntax));
    /// # }
    /// ```
    /// Use instead:
    /// ```rust,no_run
    /// # #![feature(asm)]
    /// # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// # unsafe { let ptr = "".as_ptr();
    /// # use std::arch::asm;
    /// asm!("lea {}, [{}]", lateout(reg) _, in(reg) ptr);
    /// # }
    /// ```
    #[clippy::version = "1.49.0"]
    pub INLINE_ASM_X86_ATT_SYNTAX,
    restriction,
    "prefer Intel x86 assembly syntax"
}

declare_lint_pass!(InlineAsmX86AttSyntax => [INLINE_ASM_X86_ATT_SYNTAX]);

impl EarlyLintPass for InlineAsmX86AttSyntax {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::InlineAsm(inline_asm) = &expr.kind {
            check_asm_syntax(INLINE_ASM_X86_ATT_SYNTAX, cx, inline_asm, expr.span, AsmStyle::Att);
        }
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if let ItemKind::GlobalAsm(inline_asm) = &item.kind {
            check_asm_syntax(INLINE_ASM_X86_ATT_SYNTAX, cx, inline_asm, item.span, AsmStyle::Att);
        }
    }
}
