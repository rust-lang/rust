use rustc_hir as hir;
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::ClashingFunctionNamesWithFundamentalFunctions;
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `clashing_function_names_with_fundamental_functions` lint checks for function
    /// name whose name clash with a fundamental functions expected by `core` and `std`.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(clashing_function_names_with_fundamental_functions)]
    ///
    /// #[unsafe(no_mangle)]
    /// pub fn strlen() {} // clash with the libc `strlen` function
    ///                    // care must be taken when implementing this function
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Up-most care is required when overriding those fundamental functions assumed and
    /// used by the standard library. They must follow the C specification, not use any
    /// standard-library facility or undefined behavior may occur.
    ///
    /// The symbols currently checked are respectively:
    ///  - from `core`[^1]: `memcpy`, `memmove`, `memset`, `memcmp`, `bcmp`, `strlen`
    ///  - from `std`: `read`, `write`, `open`, `close`
    ///
    /// [^1]: https://doc.rust-lang.org/core/index.html#how-to-use-the-core-library
    pub CLASHING_FUNCTION_NAMES_WITH_FUNDAMENTAL_FUNCTIONS,
    Warn,
    "using a function name that clashes with fundamental function names"
}

declare_lint_pass!(FundamentalFunctions => [CLASHING_FUNCTION_NAMES_WITH_FUNDAMENTAL_FUNCTIONS]);

static CORE_FUNDAMENTAL_FUNCTION_NAMES: &[&str] =
    &["memcpy", "memmove", "memset", "memcmp", "bcmp", "strlen"];

static STD_FUNDAMENTAL_FUNCTION_NAMES: &[&str] = &["open", "read", "write", "close"];

impl<'tcx> LateLintPass<'tcx> for FundamentalFunctions {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        let hir::ItemKind::Fn { sig: _, ident: _, generics: _, body: _, has_body: true } =
            item.kind
        else {
            return;
        };

        let Some(symbol_name) = rustc_symbol_mangling::symbol_name_without_mangling(
            cx.tcx,
            rustc_middle::ty::InstanceKind::Item(item.owner_id.to_def_id()),
        ) else {
            return;
        };

        if CORE_FUNDAMENTAL_FUNCTION_NAMES.contains(&&*symbol_name)
            || STD_FUNDAMENTAL_FUNCTION_NAMES.contains(&&*symbol_name)
        {
            cx.emit_span_lint(
                CLASHING_FUNCTION_NAMES_WITH_FUNDAMENTAL_FUNCTIONS,
                item.span,
                ClashingFunctionNamesWithFundamentalFunctions { symbol_name },
            );
        }
    }
}
