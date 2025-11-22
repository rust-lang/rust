use rustc_hir as hir;
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::RedefiningRuntimeSymbolsDiag;
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `redefining_runtime_symbols` lint checks for items whose symbol name redefines
    /// a runtime symbols expected by `core` and/or `std`.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(redefining_runtime_symbols)]
    ///
    /// #[unsafe(no_mangle)]
    /// pub fn strlen() {} // redefines the libc `strlen` function
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Up-most care is required when redefining runtime symbols assumed and
    /// used by the standard library. They must follow the C specification, not use any
    /// standard-library facility or undefined behavior may occur.
    ///
    /// The symbols currently checked are respectively:
    ///  - from `core`[^1]: `memcpy`, `memmove`, `memset`, `memcmp`, `bcmp`, `strlen`
    ///  - from `std`: `open`/`open64`, `read`, `write`, `close`
    ///
    /// [^1]: https://doc.rust-lang.org/core/index.html#how-to-use-the-core-library
    pub REDEFINING_RUNTIME_SYMBOLS,
    Warn,
    "redefining a symbol used by the standard library"
}

declare_lint_pass!(RedefiningRuntimeSymbols => [REDEFINING_RUNTIME_SYMBOLS]);

static CORE_RUNTIME_SYMBOLS: &[&str] = &["memcpy", "memmove", "memset", "memcmp", "bcmp", "strlen"];

static STD_RUNTIME_SYMBOLS: &[&str] = &["open", "open64", "read", "write", "close"];

impl<'tcx> LateLintPass<'tcx> for RedefiningRuntimeSymbols {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        // Bail-out if the item is not a function/method or static.
        match item.kind {
            hir::ItemKind::Fn { sig: _, ident: _, generics: _, body: _, has_body: true }
            | hir::ItemKind::Static(..) => {}
            _ => return,
        }

        // Compute the symbol name of our item (without mangling, as our mangling cannot ever
        // conflict with runtime symbols).
        let Some(symbol_name) = rustc_symbol_mangling::symbol_name_from_attrs(
            cx.tcx,
            rustc_middle::ty::InstanceKind::Item(item.owner_id.to_def_id()),
        ) else {
            return;
        };

        if CORE_RUNTIME_SYMBOLS.contains(&&*symbol_name)
            || STD_RUNTIME_SYMBOLS.contains(&&*symbol_name)
        {
            cx.emit_span_lint(
                REDEFINING_RUNTIME_SYMBOLS,
                item.span,
                RedefiningRuntimeSymbolsDiag { symbol_name },
            );
        }
    }
}
