use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{BytePos, Pos};
use rustc_target::spec::abi::Abi;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for Rust ABI functions with the `#[no_mangle]` attribute.
    ///
    /// ### Why is this bad?
    /// The Rust ABI is not stable, but in many simple cases matches
    /// enough with the C ABI that it is possible to forget to add
    /// `extern "C"` to a function called from C. Changes to the
    /// Rust ABI can break this at any point.
    ///
    /// ### Example
    /// ```rust
    ///  #[no_mangle]
    ///  fn example(arg_one: u32, arg_two: usize) {}
    /// ```
    ///
    /// Use instead:
    /// ```rust
    ///  #[no_mangle]
    ///  extern "C" fn example(arg_one: u32, arg_two: usize) {}
    /// ```
    #[clippy::version = "1.69.0"]
    pub NO_MANGLE_WITH_RUST_ABI,
    pedantic,
    "convert Rust ABI functions to C ABI"
}
declare_lint_pass!(NoMangleWithRustAbi => [NO_MANGLE_WITH_RUST_ABI]);

impl<'tcx> LateLintPass<'tcx> for NoMangleWithRustAbi {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Fn(fn_sig, _, _) = &item.kind {
            let attrs = cx.tcx.hir().attrs(item.hir_id());
            let mut app = Applicability::MaybeIncorrect;
            let snippet = snippet_with_applicability(cx, fn_sig.span, "..", &mut app);
            for attr in attrs {
                if let Some(ident) = attr.ident()
                    && ident.name == rustc_span::sym::no_mangle
                    && fn_sig.header.abi == Abi::Rust
                    && let Some((fn_attrs, _)) = snippet.split_once("fn")
                    && !fn_attrs.contains("extern")
                {
                    let sugg_span = fn_sig.span
                        .with_lo(fn_sig.span.lo() + BytePos::from_usize(fn_attrs.len()))
                        .shrink_to_lo();

                    span_lint_and_then(
                        cx,
                        NO_MANGLE_WITH_RUST_ABI,
                        fn_sig.span,
                        "`#[no_mangle]` set on a function with the default (`Rust`) ABI",
                        |diag| {
                            diag.span_suggestion(sugg_span, "set an ABI", "extern \"C\" ", app)
                                .span_suggestion(sugg_span, "or explicitly set the default", "extern \"Rust\" ", app);
                        },
                    );
                }
            }
        }
    }
}
