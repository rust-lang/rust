use std::fmt;

use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::ast::{Expr, ExprKind, InlineAsmOptions};
use rustc_lint::{EarlyContext, EarlyLintPass, Lint};
use rustc_session::{declare_lint_pass, declare_tool_lint};

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

fn check_expr_asm_syntax(lint: &'static Lint, cx: &EarlyContext<'_>, expr: &Expr, check_for: AsmStyle) {
    if let ExprKind::InlineAsm(ref inline_asm) = expr.kind {
        let style = if inline_asm.options.contains(InlineAsmOptions::ATT_SYNTAX) {
            AsmStyle::Att
        } else {
            AsmStyle::Intel
        };

        if style == check_for {
            span_lint_and_help(
                cx,
                lint,
                expr.span,
                &format!("{} x86 assembly syntax used", style),
                None,
                &format!("use {} x86 assembly syntax", !style),
            );
        }
    }
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of Intel x86 assembly syntax.
    ///
    /// ### Why is this bad?
    /// The lint has been enabled to indicate a preference
    /// for AT&T x86 assembly syntax.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// # #![feature(asm)]
    /// # unsafe { let ptr = "".as_ptr();
    /// # use std::arch::asm;
    /// asm!("lea {}, [{}]", lateout(reg) _, in(reg) ptr);
    /// # }
    /// ```
    /// Use instead:
    /// ```rust,no_run
    /// # #![feature(asm)]
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
        check_expr_asm_syntax(Self::get_lints()[0], cx, expr, AsmStyle::Intel);
    }
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of AT&T x86 assembly syntax.
    ///
    /// ### Why is this bad?
    /// The lint has been enabled to indicate a preference
    /// for Intel x86 assembly syntax.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// # #![feature(asm)]
    /// # unsafe { let ptr = "".as_ptr();
    /// # use std::arch::asm;
    /// asm!("lea ({}), {}", in(reg) ptr, lateout(reg) _, options(att_syntax));
    /// # }
    /// ```
    /// Use instead:
    /// ```rust,no_run
    /// # #![feature(asm)]
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
        check_expr_asm_syntax(Self::get_lints()[0], cx, expr, AsmStyle::Att);
    }
}
